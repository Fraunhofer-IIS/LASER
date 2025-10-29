#!/bin/bash

cd analysis/

python -m venv analysis-env
source analysis-env/bin/activate
pip install -r requirements.txt

