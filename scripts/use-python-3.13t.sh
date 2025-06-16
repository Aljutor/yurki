#!/bin/bash

echo "Switching to Python 3.13t (free-threaded)..."
echo "3.13t" > .python-version
uv venv --managed-python
uv sync --dev
echo "Environment ready with Python 3.13t (free-threaded)"
