#!/usr/bin/env bash
# PolicyPal AI — Launcher Script
# ==============================================================================

set -e
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Check / create virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment in .venv..."
    python3 -m venv .venv
    .venv/bin/pip install -r requirements.txt
fi

PYTHON=".venv/bin/python"
UVICORN=".venv/bin/uvicorn"

MODE="${1:-app}"

case "$MODE" in
    app)
        echo "=================================================="
        echo "  Starting PolicyPal AI — Main Web Application"
        echo "  Access UI at: http://localhost:8080"
        echo "=================================================="
        exec $UVICORN app:app --host 0.0.0.0 --port 8080
        ;;

    all|services)
        echo "=================================================="
        echo "  Starting PolicyPal AI — All Microservices & App"
        echo "  - Main UI & API:        http://localhost:8080"
        echo "  - Retrieval Service:    http://localhost:8001"
        echo "  - LLM Service:          http://localhost:8002"
        echo "  - Orchestration:        http://localhost:8000"
        echo "=================================================="
        
        $UVICORN retrieval_service:app --host 0.0.0.0 --port 8001 &
        PID_RET=$!
        $UVICORN llm_service:app --host 0.0.0.0 --port 8002 &
        PID_LLM=$!
        $UVICORN orchestration_service:app --host 0.0.0.0 --port 8000 &
        PID_ORCH=$!

        trap "kill $PID_RET $PID_LLM $PID_ORCH 2>/dev/null || true" EXIT INT TERM

        $UVICORN app:app --host 0.0.0.0 --port 8080
        ;;

    eval|evaluate)
        echo "Running PolicyPal Quantitative Evaluation..."
        $PYTHON run_evaluation.py
        ;;

    eval-week5)
        echo "Running PolicyPal Week 5 Guardrails Evaluation..."
        $PYTHON run_week5_evaluation.py
        ;;

    chunk)
        echo "Chunking PDF documents in Dataset/pdfs..."
        $PYTHON chunk_documents.py
        ;;

    embeddings)
        echo "Generating embeddings from chunks.json..."
        $PYTHON create_embeddings.py
        ;;

    *)
        echo "Usage: ./start.sh [app|all|eval|eval-week5|chunk|embeddings]"
        exit 1
        ;;
esac
