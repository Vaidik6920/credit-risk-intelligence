# Credit Risk Intelligence Platform — Makefile

.PHONY: setup data eda features train serve docker test lint clean

# ── Environment ──────────────────────────────────────────────────────────────
setup:
	pip install -r requirements.txt
	mkdir -p data/raw data/processed models mlruns data/raw/plots

# ── Data ─────────────────────────────────────────────────────────────────────
data:
	@echo "Download dataset from Kaggle:"
	@echo "  kaggle competitions download -c home-credit-default-risk"
	@echo "  unzip home-credit-default-risk.zip -d data/raw/"

# ── EDA ──────────────────────────────────────────────────────────────────────
eda:
	mkdir -p data/raw/plots
	jupyter nbconvert --to notebook --execute notebooks/01_EDA_Credit_Risk.py \
		--output notebooks/01_EDA_Credit_Risk_executed.ipynb

# ── Feature Engineering ───────────────────────────────────────────────────────
features:
	python src/feature_engineering.py

# ── Training ─────────────────────────────────────────────────────────────────
train:
	python src/train.py

# ── MLflow UI ────────────────────────────────────────────────────────────────
mlflow-ui:
	mlflow ui --host 0.0.0.0 --port 5000

# ── API (local) ───────────────────────────────────────────────────────────────
serve:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# ── Docker ───────────────────────────────────────────────────────────────────
docker-build:
	docker build -t credit-risk-api:latest .

docker-run:
	docker run -p 8000:8000 credit-risk-api:latest

docker-compose-up:
	docker-compose up --build

# ── Testing ──────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short

# ── Lint ─────────────────────────────────────────────────────────────────────
lint:
	flake8 src/ api/ --max-line-length=100
	black src/ api/ --check

format:
	black src/ api/

# ── Clean ────────────────────────────────────────────────────────────────────
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf data/processed/*.parquet
	rm -rf models/*.pkl

# ── Full Pipeline ─────────────────────────────────────────────────────────────
all: features train serve
