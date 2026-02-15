#!/bin/bash

echo "=========================================="
echo "Starting ML Classification App"
echo "=========================================="

cd /Users/vinoth-5221/Desktop/ML

# Check if installation is complete
if [ -f "venv/bin/streamlit" ]; then
    echo "✅ Dependencies installed"
    echo "🚀 Starting Streamlit app..."
    echo ""
    ./venv/bin/streamlit run app.py
else
    echo "⏳ Installing dependencies (this may take 2-3 minutes)..."
    ./venv/bin/pip install -r requirements.txt
    echo ""
    echo "✅ Installation complete!"
    echo "🚀 Starting Streamlit app..."
    echo ""
    ./venv/bin/streamlit run app.py
fi
