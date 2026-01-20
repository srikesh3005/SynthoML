# Synthetic Data Generator

Generate synthetic datasets using a web interface. Upload your CSV, train a model, and download synthetic data.

## Quick Start

**Terminal 1 - Backend:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python server.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm install
npm run dev
```

**Open:** http://localhost:5173

## Usage

1. Upload your CSV file
2. Set number of training epochs
3. Click "Train Model"
4. Generate synthetic data
5. Download results

---

> **Python 3.14 Note**: Uses statistical fallback generator (CTGAN requires Python ≤3.13)
