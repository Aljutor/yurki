#!/bin/bash

echo "Switching to Python 3.12..."
echo "3.12" > .python-version
uv venv --managed-python
uv sync --dev
echo "Environment ready with Python 3.12"
