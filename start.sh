#!/bin/bash

# Ensure we are in the project root
cd "$(dirname "$0")"

# 1. Set SSL Certificates (Fixes the download error)
export SSL_CERT_FILE=$(python3 -c "import certifi; print(certifi.where())" 2>/dev/null)
export REQUESTS_CA_BUNDLE=$SSL_CERT_FILE

# 2. Check for .env file
if [ ! -f ".env" ]; then
    echo "Creating .env from .env.example..."
    cp .env.example .env
fi

# 3. Navigate to backend
cd backend

# 4. Handle Virtual Environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# 5. Pre-warm AI models (Download them now so the web app is fast)
echo "Checking AI models (this may take a minute on first run)..."
python scripts/prewarm_models.py

# 6. Start the server
echo "----------------------------------------------------"
echo "Starting DeafAccessibility Assistant..."
echo "Open: http://localhost:5051"
echo "----------------------------------------------------"
python server.py
