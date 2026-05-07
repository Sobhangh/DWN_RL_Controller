#!/usr/bin/env bash
set -e  # exit on first error

# the Python interpreter we want
PYTHON=python3.9

# make sure python3.9 exists
command -v "$PYTHON" >/dev/null 2>&1 \
  || { echo "❌ $PYTHON not found; please install it"; exit 1; }

# install uv if missing
command -v uv >/dev/null 2>&1 || pip install uv

VENV="difflogic_verification"

# create (or reuse) the venv with python3.9
uv venv --python "$PYTHON" "$VENV"

# activate it
source "$VENV/bin/activate"

# now install deps inside that venv
uv pip install -r requirements.txt

# ... rest of your kissat build logic unchanged ...
if ! [ -x kissat/build/kissat ]; then
    echo "kissat not found, building kissat"
    git clone https://github.com/arminbiere/kissat.git
    cd kissat
    ./configure
    make
    cd ..
fi

echo "✅ Setup complete. Run 'source $VENV/bin/activate'"
