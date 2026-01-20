#!/bin/bash

# 1. Run the model training script
echo "========================================"
echo "Step 1: Training Model (src/model.py)..."
echo "========================================"
python3 -m src.model

# 2. Start the Streamlit application
echo ""
echo "========================================"
echo "Step 2: Starting Application (app.py)..."
echo "========================================"
# Check if streamlit is running and kill it to avoid port conflict
pkill -f streamlit
streamlit run app.py
