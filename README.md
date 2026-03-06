# AI API Skeleton (FastAPI + Docker)

## Run locally
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

## Run with Docker
```bash
docker build -t ai-api:0.1.0 .
docker run --rm -p 8000:8000 ai-api:0.1.0
```

## Endpoints
* GET /health
* GET /version
* POST /predict