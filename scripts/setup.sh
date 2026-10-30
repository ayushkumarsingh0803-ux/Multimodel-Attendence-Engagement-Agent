#!/bin/bash
# ============================================================
# setup.sh — Environment setup for Multimodal Attendance Agent
# ============================================================

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$PROJECT_DIR/venv"

echo ""
echo "🎓 Multimodal Attendance & Engagement Agent — Setup"
echo "====================================================="
echo "Project directory: $PROJECT_DIR"
echo ""

# ── 1. Python version check ───────────────────────────────────────────────────
echo "→ Checking Python version..."
python3 --version 2>/dev/null || { echo "❌ Python 3 is required but not found."; exit 1; }
PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(echo $PY_VER | cut -d. -f1)
PY_MINOR=$(echo $PY_VER | cut -d. -f2)

if [ "$PY_MAJOR" -lt 3 ] || ([ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]); then
    echo "❌ Python 3.10+ is required (found $PY_VER)."
    exit 1
fi
echo "  ✓ Python $PY_VER"

# ── 2. Create virtual environment ─────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "→ Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo "  ✓ Virtual environment created at $VENV_DIR"
else
    echo "  ✓ Virtual environment already exists."
fi

# ── 3. Activate and install dependencies ──────────────────────────────────────
echo "→ Installing dependencies..."
source "$VENV_DIR/bin/activate"
pip install --upgrade pip --quiet
pip install -r "$PROJECT_DIR/requirements.txt" --quiet
echo "  ✓ Dependencies installed."

# ── 4. Create required directories ────────────────────────────────────────────
echo "→ Creating project directories..."
mkdir -p "$PROJECT_DIR/uploads"
mkdir -p "$PROJECT_DIR/generated_audio"
mkdir -p "$PROJECT_DIR/generated_code"
echo "  ✓ Directories ready."

# ── 5. Copy .env.example to .env if not present ───────────────────────────────
if [ ! -f "$PROJECT_DIR/.env" ]; then
    cp "$PROJECT_DIR/.env.example" "$PROJECT_DIR/.env"
    echo "  ✓ .env created from .env.example"
    echo ""
    echo "  ⚠️  IMPORTANT: Edit .env and add your GEMINI_API_KEY before running."
else
    echo "  ✓ .env already exists."
fi

# ── 6. Initialise database ────────────────────────────────────────────────────
echo "→ Initialising database..."
cd "$PROJECT_DIR"
python3 -c "
from app import create_app
from config import Config
app = create_app(Config)
print('  ✓ Database initialised.')
"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Edit .env and set your GEMINI_API_KEY"
echo "  2. Activate the venv: source venv/bin/activate"
echo "  3. Run the app:       python scripts/run.py"
echo "  4. Open in browser:   http://localhost:5000"
echo ""
