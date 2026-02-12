#!/bin/bash
# Ustawia ścieżkę, żeby Python widział folder src jako moduł
export PYTHONPATH=$PYTHONPATH:.

echo "Uruchamiam trening..."
python run_pipeline.py