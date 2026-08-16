#!/bin/bash

# Launch this script with: source venv_install.sh
# This script sets up a Python virtual environment
# and installs the required dependencies.
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip

python3 -m pip install -r requirements.txt