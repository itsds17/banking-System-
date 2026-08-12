# ==============================================================================
# Makefile — Banking Decision Intelligence Platform
# Common developer commands. Run `make help` to see all targets.
# ==============================================================================

.PHONY: help setup venv install generate-data docker-up docker-down \
        test test-cov lint clean mlflow-ui api-serve dashboard

# ── Config ────────────────────────────────────────────────────────────────────
PYTHON      := python
PIP         := pip
VENV        := .venv
VENV_BIN    := $(VENV)/Scripts
DATA_CFG    := config/data_config.yaml

# ── Help ──────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  Banking Decision Intelligence — Developer Commands"
	@echo "  =================================================="
	@echo "  make setup          Create venv + install all dependencies"
	@echo "  make install        Install dependencies into existing venv"
	@echo "  make generate-data  Generate synthetic banking dataset"
	@echo "  make docker-up      Start all Docker services (PG, Kafka, MLflow)"
	@echo "  make docker-down    Stop all Docker services"
	@echo "  make docker-logs    Tail Docker service logs"
	@echo "  make test           Run all pytest tests"
	@echo "  make test-cov       Run tests with HTML coverage report"
	@echo "  make lint           Run ruff linter"
	@echo "  make mlflow-ui      Open MLflow tracking UI"
	@echo "  make api-serve      Start FastAPI server (development)"
	@echo "  make dashboard      Start Streamlit dashboard"
	@echo "  make clean          Remove generated data, cache, and build files"
	@echo ""

# ── Setup ─────────────────────────────────────────────────────────────────────
setup: venv install
	@echo "Setup complete."

venv:
	$(PYTHON) -m venv $(VENV)
	@echo "Virtual environment created at $(VENV)"

install:
	$(VENV_BIN)/python -m pip install --upgrade pip
	$(VENV_BIN)/pip install -r requirements.txt
	@echo "Dependencies installed."

# ── Data ──────────────────────────────────────────────────────────────────────
generate-data:
	$(VENV_BIN)/python scripts/generate_data.py --config $(DATA_CFG)

ingest-data:
	$(VENV_BIN)/python scripts/ingest_data.py

build-features:
	$(VENV_BIN)/python scripts/build_features.py

generate-data-small:
	$(VENV_BIN)/python scripts/generate_data.py --config $(DATA_CFG) \
		--customers 1000 --transactions 50000

generate-data-large:
	$(VENV_BIN)/python scripts/generate_data.py --config $(DATA_CFG) \
		--customers 50000 --transactions 2000000

# ── Docker ────────────────────────────────────────────────────────────────────
docker-up:
	docker compose up -d
	@echo "Services started. Access:"
	@echo "  PostgreSQL : localhost:5432"
	@echo "  Kafka      : localhost:9092"
	@echo "  MLflow UI  : http://localhost:5000"

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

docker-ps:
	docker compose ps

# ── Testing ───────────────────────────────────────────────────────────────────
test:
	$(VENV_BIN)/pytest tests/ -v --tb=short

test-cov:
	$(VENV_BIN)/pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing
	@echo "Coverage report: htmlcov/index.html"

test-fast:
	$(VENV_BIN)/pytest tests/ -v --tb=short -m "not slow"

# ── Linting ───────────────────────────────────────────────────────────────────
lint:
	$(VENV_BIN)/ruff check src/ tests/ scripts/ api/

format:
	$(VENV_BIN)/ruff format src/ tests/ scripts/ api/

# ── Services ──────────────────────────────────────────────────────────────────
mlflow-ui:
	start http://localhost:5000

api-serve:
	$(VENV_BIN)/uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	$(VENV_BIN)/streamlit run src/monitoring/dashboard.py

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	del /Q /S data\synthetic\*.parquet data\synthetic\*.csv 2>nul || true
	del /Q /S data\processed\*.parquet data\processed\*.csv 2>nul || true
	rmdir /S /Q htmlcov 2>nul || true
	rmdir /S /Q __pycache__ 2>nul || true
	rmdir /S /Q .pytest_cache 2>nul || true
	@echo "Cleaned up generated files."
