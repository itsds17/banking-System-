# ENVIRONMENT SETUP REPORT
## Real-Time Banking Decision Intelligence & Risk Analytics Platform

> Generated: 2026-08-12 | Machine: Windows 11 x64

---

## 1. DETECTED ENVIRONMENT

| Component | Detected | Status |
|---|---|---|
| **OS** | Windows 11 Home Single Language (Build 26200, x64) | Supported |
| **Python** | 3.9.12 (C:\Python39) | Meets minimum |
| **pip** | 25.0.1 | OK |
| **Java / JDK** | OpenJDK 17.0.10 (Amazon Corretto-17.0.10.7.1, 64-bit) | Required for PySpark |
| **Git** | 2.53.0.windows.1 | OK |
| **Docker CLI** | 28.5.2 | Installed |
| **Docker Compose** | v2.40.3-desktop.1 | Installed |
| **Docker Daemon** | NOT RUNNING | MUST be started |
| **RAM** | 8 GB total | Tight - see notes |
| **Disk (C:)** | 165.54 GB free (of ~475 GB) | Sufficient |
| **Git Repo (local)** | Not initialised in workspace | Must be initialised |
| **Git global user.name** | Not configured | Should be configured |
| **Git global user.email** | Not configured | Should be configured |

---

## 2. PORT AVAILABILITY

All required ports are currently AVAILABLE (nothing is using them):

| Port | Service | Status |
|---|---|---|
| 5432 | PostgreSQL | Free |
| 9092 | Kafka Broker | Free |
| 2181 | Zookeeper | Free |
| 5000 | MLflow Tracking UI | Free |
| 8000 | FastAPI Model Serving | Free |
| 8080 | Alternative / Admin | Free |
| 8501 | Streamlit Dashboard | Free |

---

## 3. PYTHON PACKAGES - ALREADY INSTALLED

| Package | Version | Project Use |
|---|---|---|
| scikit-learn | 1.6.1 | Fraud + credit-risk models, clustering |
| numpy | 2.0.2 | Numerical computing |
| pandas | 2.2.3 | DataFrame processing |
| scipy | 1.13.1 | Statistical analysis |
| networkx | 3.2.1 | Graph fraud analytics |
| matplotlib | 3.9.4 | Visualisations |
| seaborn | 0.13.2 | Statistical plots |
| xgboost | 1.7.6 | Gradient-boosted fraud + credit models |
| fastapi | 0.104.1 | Model serving API |
| streamlit | 1.41.1 | Dashboard |

---

## 4. PYTHON PACKAGES - MISSING (MUST INSTALL)

| Package | Required Version | Purpose |
|---|---|---|
| **pyspark** | 3.5.x | Distributed batch processing |
| **kafka-python** | 2.0.x | Kafka producer / consumer |
| **mlflow** | 2.x | Experiment tracking & model registry |
| **shap** | 0.44.x | Explainable AI (SHAP values) |
| **plotly** | 5.x | Interactive charts in dashboard |
| **psycopg2-binary** | 2.9.x | PostgreSQL connector |
| **sqlalchemy** | 2.x | SQL ORM for feature pipelines |
| **faker** | 24.x | Synthetic data generation |
| **uvicorn** | 0.x | ASGI server for FastAPI |
| **httpx** | 0.x | API test client |
| **pytest** | 7.x | Unit & integration testing |
| **pytest-cov** | 4.x | Test coverage |
| **python-dotenv** | 1.x | Environment variable loading |
| **great-expectations** | 0.18.x | Data quality validation |
| **statsmodels** | 0.14.x | Statistical modelling / forecasting |
| **imbalanced-learn** | 0.12.x | Class-imbalance handling (SMOTE etc.) |

> NOTE: There is a partially-corrupted distribution (-ransformers) in your site-packages.
> It is harmless but shows a pip WARNING on every pip show command.
> It can be cleaned up with: pip uninstall transformers --yes (only if you confirm it is unused).

---

## 5. DOCKER STATUS - ACTION REQUIRED

Docker Desktop is INSTALLED but the DAEMON IS NOT RUNNING.

The error received:
```
error during connect: open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified.
```

This means Docker Desktop is not currently open.

### Fix:
1. Open Docker Desktop from the Start Menu or taskbar.
2. Wait for it to fully start (the Docker whale icon in the taskbar should show green / "Running").
3. Run `docker info` again to confirm the daemon is available.

> All project services (PostgreSQL, Kafka, MLflow, FastAPI) run inside Docker containers.
> Docker must be running before Phase 14 (Dockerisation) but is NOT needed for Phases 1-3.

---

## 6. GIT REPOSITORY STATUS - ACTION REQUIRED

### Local:
The workspace folder `c:\Users\divya\OneDrive\Desktop\Bnaking System` is NOT a Git repository.
It needs to be initialised and connected to your GitHub repo.

### GitHub repo provided:
```
https://github.com/itsds17/banking-System-
```

### Fix - run these commands once you confirm Git user details:
```powershell
# Step 1 - Set Git identity (replace with your real details)
git config --global user.name "Your Name"
git config --global user.email "your-email@example.com"

# Step 2 - Initialise the repo inside the project folder
cd "c:\Users\divya\OneDrive\Desktop\Bnaking System"
git init
git remote add origin https://github.com/itsds17/banking-System-.git
git branch -M main
```

> We will do this as part of Phase 1 after you confirm your Git username and email.

---

## 7. RESOURCE ASSESSMENT & CONFIGURATION RECOMMENDATIONS

### RAM: 8 GB (Tight but workable)

Recommended Docker memory caps:

| Service | Memory Cap |
|---|---|
| PostgreSQL (Docker) | 512 MB |
| Kafka + Zookeeper (Docker) | 1 GB total |
| MLflow (Docker) | 256 MB |
| FastAPI (Docker) | 256 MB |
| PySpark (local JVM) | 2 GB (--driver-memory 2g) |
| Streamlit | ~200 MB |
| Available for OS + other apps | ~3.8 GB |

Recommended synthetic data scale for 8 GB machine:

| Entity | Default Count | Max (configurable) |
|---|---|---|
| Customers | 10,000 | 50,000 |
| Accounts | 20,000 | 100,000 |
| Transactions | 500,000 | 2,000,000+ |
| Devices | 5,000 | 20,000 |
| Merchants | 2,000 | 10,000 |
| IP Addresses | 3,000 | 10,000 |
| Loans | 6,000 | 30,000 |

> The data generator will be fully configurable via config/data_config.yaml.
> You can scale up by closing other applications or upgrading RAM.

### Disk: 165 GB free - No concern

| Item | Estimate |
|---|---|
| Docker images (PG, Kafka, MLflow) | ~3-5 GB |
| Synthetic data files (CSV/Parquet) | ~500 MB - 2 GB |
| Python virtual environment | ~1-2 GB |
| Model artefacts + MLflow runs | ~200 MB |
| Logs | ~100 MB |
| Total | ~5-10 GB |

---

## 8. PYTHON VERSION COMPATIBILITY

Python 3.9.12 is compatible with all required libraries:

| Library | Python 3.9 Support |
|---|---|
| PySpark 3.5.x | Compatible |
| MLflow 2.x | Compatible |
| SHAP 0.44.x | Compatible |
| XGBoost 1.7.6 | Compatible |
| FastAPI 0.104.1 | Compatible |

Recommendation: Use a Python virtual environment (venv) for this project
to isolate dependencies from your global Python installation.
We will create one as part of Phase 1.

---

## 9. JAVA VERSION NOTE

Java 17 (Amazon Corretto) is installed.
PySpark 3.5.x requires Java 8, 11, or 17. Java 17 is FULLY SUPPORTED.

The JAVA_HOME environment variable must point to the JDK installation.

Verify it is set:
```powershell
echo $env:JAVA_HOME
```

If it is empty, set it to the Corretto installation directory (typically):
```powershell
$env:JAVA_HOME = "C:\Program Files\Amazon Corretto\jdk17.0.10_7"
```

---

## 10. REQUIRED ACTIONS BEFORE PHASE 1

| # | Action | Priority |
|---|---|---|
| 1 | Open Docker Desktop and wait for daemon to start | CRITICAL |
| 2 | Confirm Git username and email so we can initialise the repo | CRITICAL |
| 3 | Verify JAVA_HOME is set for PySpark | IMPORTANT |
| 4 | Optionally clean up broken -ransformers pip entry | OPTIONAL |

---

## 11. INSTALLATION INSTRUCTIONS (Phase 1 - run after approval)

### Step 1: Create virtual environment
```powershell
cd "c:\Users\divya\OneDrive\Desktop\Bnaking System"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 2: Upgrade pip
```powershell
python -m pip install --upgrade pip
```

### Step 3: Install all missing packages
```powershell
pip install pyspark==3.5.1 kafka-python==2.0.2 mlflow==2.11.1 shap==0.44.1 `
    plotly==5.20.0 psycopg2-binary==2.9.9 sqlalchemy==2.0.28 faker==24.3.0 `
    uvicorn==0.27.1 httpx==0.27.0 pytest==7.4.4 pytest-cov==4.1.0 `
    python-dotenv==1.0.1 imbalanced-learn==0.12.2 statsmodels==0.14.1 `
    great-expectations==0.18.12 pyarrow==15.0.0
```

> All packages are open-source. No paid APIs or cloud accounts required.

---

## 12. SUMMARY TABLE

| Category | Status |
|---|---|
| Python 3.9.12 | READY |
| Java 17 (Corretto) | READY |
| Git 2.53 | READY |
| Docker CLI | INSTALLED |
| Docker Daemon | START DOCKER DESKTOP |
| Git repo (local) | NEEDS git init |
| Git identity | NEEDS user.name / user.email |
| Core ML packages | READY (scikit-learn, xgboost, pandas, numpy) |
| Streaming/DB packages | MISSING (kafka-python, psycopg2) |
| MLOps packages | MISSING (mlflow, shap) |
| PySpark | NOT INSTALLED |
| All ports | ALL FREE |
| Disk space | 165 GB FREE - SUFFICIENT |
| RAM | 8 GB - MANAGEABLE WITH CONFIG |

---

*This file was auto-generated during Phase 0 environment inspection.*
*It will be committed to the GitHub repository as part of Phase 1.*
