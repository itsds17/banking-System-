-- ==============================================================================
-- SQL Feature Engineering & Risk Analytics Views
-- Real-Time Banking Decision Intelligence Platform
-- Phase 2 Feature Pipeline
-- ==============================================================================

SET search_path TO banking, public;

-- ── 1. Customer Spending & Behavioural Features ───────────────────────────────
CREATE OR REPLACE VIEW banking.v_customer_features AS
SELECT
    c.customer_id,
    c.age,
    c.gender,
    c.income,
    c.employment_status,
    c.city,
    c.state,
    c.credit_score,
    c.risk_profile,
    COUNT(t.transaction_id)                         AS total_transactions,
    COALESCE(SUM(t.amount), 0)                      AS total_spend,
    COALESCE(AVG(t.amount), 0)                      AS avg_transaction_amount,
    COALESCE(STDDEV(t.amount), 0)                   AS stddev_transaction_amount,
    COALESCE(MAX(t.amount), 0)                      AS max_transaction_amount,
    MAX(t.timestamp)                                AS last_transaction_timestamp,
    COUNT(DISTINCT t.device_id)                     AS unique_devices_used,
    COUNT(DISTINCT t.ip_address)                    AS unique_ips_used,
    COUNT(DISTINCT t.merchant_id)                   AS unique_merchants_visited,
    SUM(CASE WHEN t.is_fraud THEN 1 ELSE 0 END)    AS fraud_transaction_count,
    CASE WHEN COUNT(t.transaction_id) > 0
         THEN ROUND(SUM(CASE WHEN t.is_fraud THEN 1 ELSE 0 END)::NUMERIC / COUNT(t.transaction_id), 4)
         ELSE 0
    END                                             AS customer_fraud_rate
FROM banking.customers c
LEFT JOIN banking.transactions t ON c.customer_id = t.customer_id
GROUP BY c.customer_id, c.age, c.gender, c.income, c.employment_status,
         c.city, c.state, c.credit_score, c.risk_profile;

-- ── 2. Device Intelligence & Shared Device Risk ───────────────────────────────
CREATE OR REPLACE VIEW banking.v_device_intelligence AS
SELECT
    d.device_id,
    d.device_type,
    d.operating_system,
    COUNT(DISTINCT cd.customer_id)                  AS shared_customer_count,
    d.transaction_count                             AS global_transaction_count,
    d.fraud_count                                   AS global_fraud_count,
    CASE WHEN d.transaction_count > 0
         THEN ROUND(d.fraud_count::NUMERIC / d.transaction_count, 4)
         ELSE 0
    END                                             AS device_fraud_rate,
    CASE WHEN COUNT(DISTINCT cd.customer_id) > 3 THEN 0.85
         WHEN COUNT(DISTINCT cd.customer_id) > 1 THEN 0.40
         ELSE 0.05
    END                                             AS device_risk_score
FROM banking.devices d
LEFT JOIN banking.customer_devices cd ON d.device_id = cd.device_id
GROUP BY d.device_id, d.device_type, d.operating_system,
         d.transaction_count, d.fraud_count;

-- ── 3. IP Intelligence & Shared IP Risk ───────────────────────────────────────
CREATE OR REPLACE VIEW banking.v_ip_intelligence AS
SELECT
    ip.ip_address,
    COUNT(DISTINCT ci.customer_id)                  AS shared_customer_count,
    ip.transaction_count                            AS global_transaction_count,
    ip.fraud_count                                  AS global_fraud_count,
    CASE WHEN ip.transaction_count > 0
         THEN ROUND(ip.fraud_count::NUMERIC / ip.transaction_count, 4)
         ELSE 0
    END                                             AS ip_fraud_rate,
    CASE WHEN COUNT(DISTINCT ci.customer_id) > 5 THEN 0.90
         WHEN COUNT(DISTINCT ci.customer_id) > 2 THEN 0.50
         ELSE 0.05
    END                                             AS ip_risk_score
FROM banking.ip_addresses ip
LEFT JOIN banking.customer_ips ci ON ip.ip_address = ci.ip_address
GROUP BY ip.ip_address, ip.transaction_count, ip.fraud_count;

-- ── 4. New Device Detection View ──────────────────────────────────────────────
-- Flags incoming transactions where device_id was not previously in customer's known devices
CREATE OR REPLACE VIEW banking.v_new_device_detection AS
SELECT
    t.transaction_id,
    t.customer_id,
    t.account_id,
    t.device_id,
    t.ip_address,
    t.amount,
    t.timestamp,
    t.merchant_id,
    t.merchant_category,
    t.is_fraud,
    t.fraud_scenario,
    CASE
        WHEN cd.customer_id IS NULL THEN TRUE   -- Unrecognized/new device for this customer!
        ELSE FALSE
    END                                             AS is_new_device,
    COALESCE(cd.use_count, 0)                       AS customer_device_use_count,
    COALESCE(di.device_risk_score, 0.05)            AS device_risk_score,
    COALESCE(ii.ip_risk_score, 0.05)                AS ip_risk_score
FROM banking.transactions t
LEFT JOIN banking.customer_devices cd
       ON t.customer_id = cd.customer_id
      AND t.device_id = cd.device_id
LEFT JOIN banking.v_device_intelligence di
       ON t.device_id = di.device_id
LEFT JOIN banking.v_ip_intelligence ii
       ON t.ip_address = ii.ip_address;

-- ── 5. Transaction Velocity & Rolling Window Aggregates ───────────────────────
CREATE OR REPLACE VIEW banking.v_transaction_velocity AS
SELECT
    t.transaction_id,
    t.customer_id,
    t.timestamp,
    t.amount,
    COUNT(*) OVER (
        PARTITION BY t.customer_id
        ORDER BY t.timestamp
        RANGE BETWEEN INTERVAL '1 hour' PRECEDING AND CURRENT ROW
    ) - 1                                           AS txns_last_1_hour,
    COALESCE(SUM(t.amount) OVER (
        PARTITION BY t.customer_id
        ORDER BY t.timestamp
        RANGE BETWEEN INTERVAL '1 hour' PRECEDING AND CURRENT ROW
    ), 0)                                           AS spend_last_1_hour,
    COUNT(*) OVER (
        PARTITION BY t.customer_id
        ORDER BY t.timestamp
        RANGE BETWEEN INTERVAL '24 hours' PRECEDING AND CURRENT ROW
    ) - 1                                           AS txns_last_24_hours,
    COALESCE(SUM(t.amount) OVER (
        PARTITION BY t.customer_id
        ORDER BY t.timestamp
        RANGE BETWEEN INTERVAL '24 hours' PRECEDING AND CURRENT ROW
    ), 0)                                           AS spend_last_24_hours
FROM banking.transactions t;

-- ── 6. Stored Function for Real-Time New-Device Check ────────────────────────
CREATE OR REPLACE FUNCTION banking.fn_check_new_device(
    p_customer_id VARCHAR(20),
    p_device_id VARCHAR(40)
)
RETURNS TABLE (
    is_new_device BOOLEAN,
    previous_use_count INTEGER,
    first_used_timestamp TIMESTAMP
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        CASE WHEN cd.customer_id IS NULL THEN TRUE ELSE FALSE END AS is_new_device,
        COALESCE(cd.use_count, 0) AS previous_use_count,
        cd.first_used AS first_used_timestamp
    FROM (SELECT p_customer_id AS cid, p_device_id AS did) req
    LEFT JOIN banking.customer_devices cd
           ON req.cid = cd.customer_id
          AND req.did = cd.device_id;
END;
$$ LANGUAGE plpgsql;
