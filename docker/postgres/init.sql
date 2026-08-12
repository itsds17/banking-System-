-- ==============================================================================
-- PostgreSQL Initialisation Script
-- Banking Decision Intelligence Platform
-- Run automatically by Docker on first container start.
-- ==============================================================================

-- ── Schema ────────────────────────────────────────────────────────────────────
CREATE SCHEMA IF NOT EXISTS banking;
SET search_path TO banking, public;

-- ── Customers ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS banking.customers (
    customer_id         VARCHAR(20) PRIMARY KEY,
    age                 INTEGER NOT NULL,
    gender              VARCHAR(10),
    income              NUMERIC(12, 2),
    employment_status   VARCHAR(30),
    city                VARCHAR(60),
    state               VARCHAR(60),
    customer_since      DATE,
    credit_score        INTEGER,
    risk_profile        VARCHAR(10),
    created_at          TIMESTAMP DEFAULT NOW()
);

-- ── Accounts ──────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS banking.accounts (
    account_id          VARCHAR(20) PRIMARY KEY,
    customer_id         VARCHAR(20) NOT NULL REFERENCES banking.customers(customer_id),
    account_type        VARCHAR(20),
    balance             NUMERIC(14, 2),
    account_open_date   DATE,
    status              VARCHAR(15),
    created_at          TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_accounts_customer ON banking.accounts(customer_id);

-- ── Merchants ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS banking.merchants (
    merchant_id         VARCHAR(20) PRIMARY KEY,
    merchant_name       VARCHAR(100),
    merchant_category   VARCHAR(40),
    city                VARCHAR(60),
    state               VARCHAR(60),
    latitude            NUMERIC(9, 6),
    longitude           NUMERIC(9, 6),
    merchant_risk_score NUMERIC(5, 4),
    created_at          TIMESTAMP DEFAULT NOW()
);

-- ── Devices ───────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS banking.devices (
    device_id           VARCHAR(40) PRIMARY KEY,
    device_type         VARCHAR(20),
    operating_system    VARCHAR(20),
    first_seen          TIMESTAMP,
    last_seen           TIMESTAMP,
    transaction_count   INTEGER DEFAULT 0,
    fraud_count         INTEGER DEFAULT 0,
    created_at          TIMESTAMP DEFAULT NOW()
);

-- ── IP Addresses ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS banking.ip_addresses (
    ip_id               SERIAL PRIMARY KEY,
    ip_address          VARCHAR(45) UNIQUE NOT NULL,
    first_seen          TIMESTAMP,
    last_seen           TIMESTAMP,
    transaction_count   INTEGER DEFAULT 0,
    fraud_count         INTEGER DEFAULT 0,
    customer_count      INTEGER DEFAULT 0,
    created_at          TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ip_address ON banking.ip_addresses(ip_address);

-- ── Transactions ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS banking.transactions (
    transaction_id      VARCHAR(40) PRIMARY KEY,
    customer_id         VARCHAR(20) REFERENCES banking.customers(customer_id),
    account_id          VARCHAR(20) REFERENCES banking.accounts(account_id),
    merchant_id         VARCHAR(20) REFERENCES banking.merchants(merchant_id),
    device_id           VARCHAR(40) REFERENCES banking.devices(device_id),
    ip_address          VARCHAR(45),
    amount              NUMERIC(12, 2),
    timestamp           TIMESTAMP NOT NULL,
    transaction_type    VARCHAR(20),
    merchant_category   VARCHAR(40),
    city                VARCHAR(60),
    state               VARCHAR(60),
    latitude            NUMERIC(9, 6),
    longitude           NUMERIC(9, 6),
    payment_method      VARCHAR(30),
    is_fraud            BOOLEAN DEFAULT FALSE,
    fraud_scenario      VARCHAR(50),
    created_at          TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_txn_customer ON banking.transactions(customer_id);
CREATE INDEX IF NOT EXISTS idx_txn_timestamp ON banking.transactions(timestamp);
CREATE INDEX IF NOT EXISTS idx_txn_device ON banking.transactions(device_id);
CREATE INDEX IF NOT EXISTS idx_txn_fraud ON banking.transactions(is_fraud);

-- ── Customer–Device mapping ───────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS banking.customer_devices (
    id                  SERIAL PRIMARY KEY,
    customer_id         VARCHAR(20) REFERENCES banking.customers(customer_id),
    device_id           VARCHAR(40) REFERENCES banking.devices(device_id),
    first_used          TIMESTAMP,
    last_used           TIMESTAMP,
    use_count           INTEGER DEFAULT 1,
    UNIQUE(customer_id, device_id)
);

CREATE INDEX IF NOT EXISTS idx_cust_dev_customer ON banking.customer_devices(customer_id);
CREATE INDEX IF NOT EXISTS idx_cust_dev_device ON banking.customer_devices(device_id);

-- ── Customer–IP mapping ───────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS banking.customer_ips (
    id                  SERIAL PRIMARY KEY,
    customer_id         VARCHAR(20) REFERENCES banking.customers(customer_id),
    ip_address          VARCHAR(45),
    first_used          TIMESTAMP,
    last_used           TIMESTAMP,
    use_count           INTEGER DEFAULT 1,
    UNIQUE(customer_id, ip_address)
);

CREATE INDEX IF NOT EXISTS idx_cust_ip_customer ON banking.customer_ips(customer_id);

-- ── Loans ─────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS banking.loans (
    loan_id             VARCHAR(20) PRIMARY KEY,
    customer_id         VARCHAR(20) REFERENCES banking.customers(customer_id),
    loan_amount         NUMERIC(14, 2),
    interest_rate       NUMERIC(5, 2),
    tenure_months       INTEGER,
    monthly_income      NUMERIC(12, 2),
    debt_to_income      NUMERIC(6, 4),
    employment_status   VARCHAR(30),
    delinquency_history INTEGER DEFAULT 0,
    loan_status         VARCHAR(20),
    default_flag        BOOLEAN DEFAULT FALSE,
    loan_start_date     DATE,
    created_at          TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_loans_customer ON banking.loans(customer_id);

-- ── Useful Views ──────────────────────────────────────────────────────────────

-- Customer transaction summary
CREATE OR REPLACE VIEW banking.v_customer_transaction_summary AS
SELECT
    c.customer_id,
    COUNT(t.transaction_id)                         AS total_transactions,
    SUM(t.amount)                                   AS total_spend,
    AVG(t.amount)                                   AS avg_transaction_amount,
    STDDEV(t.amount)                                AS stddev_amount,
    MAX(t.timestamp)                                AS last_transaction_date,
    MIN(t.timestamp)                                AS first_transaction_date,
    COUNT(DISTINCT t.device_id)                     AS unique_devices,
    COUNT(DISTINCT t.ip_address)                    AS unique_ips,
    COUNT(DISTINCT t.merchant_id)                   AS unique_merchants,
    SUM(CASE WHEN t.is_fraud THEN 1 ELSE 0 END)    AS fraud_transaction_count
FROM banking.customers c
LEFT JOIN banking.transactions t ON c.customer_id = t.customer_id
GROUP BY c.customer_id;

-- Device risk view
CREATE OR REPLACE VIEW banking.v_device_risk AS
SELECT
    d.device_id,
    d.device_type,
    d.operating_system,
    COUNT(DISTINCT cd.customer_id)                  AS customer_count,
    d.transaction_count,
    d.fraud_count,
    CASE WHEN d.transaction_count > 0
         THEN ROUND(d.fraud_count::NUMERIC / d.transaction_count, 4)
         ELSE 0
    END                                             AS fraud_rate,
    EXTRACT(EPOCH FROM (NOW() - d.first_seen))/86400 AS device_age_days
FROM banking.devices d
LEFT JOIN banking.customer_devices cd ON d.device_id = cd.device_id
GROUP BY d.device_id, d.device_type, d.operating_system,
         d.transaction_count, d.fraud_count, d.first_seen;

-- High-risk customers
CREATE OR REPLACE VIEW banking.v_high_risk_customers AS
SELECT
    c.customer_id,
    c.risk_profile,
    c.credit_score,
    ts.total_transactions,
    ts.fraud_transaction_count,
    ts.unique_devices,
    CASE WHEN ts.total_transactions > 0
         THEN ROUND(ts.fraud_transaction_count::NUMERIC / ts.total_transactions, 4)
         ELSE 0
    END AS fraud_rate
FROM banking.customers c
JOIN banking.v_customer_transaction_summary ts ON c.customer_id = ts.customer_id
WHERE c.risk_profile = 'high'
   OR ts.fraud_transaction_count > 0
   OR c.credit_score < 550;

COMMIT;
