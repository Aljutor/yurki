#!/bin/bash

echo "Switching to Python 3.13..."
echo "3.13" > .python-version
uv venv --managed-python
uv sync --dev
echo "Environment ready with Python 3.13"
