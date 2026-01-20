# Synthetic Data Generator

Generate synthetic datasets using a web interface. Upload your CSV, train a model, and download synthetic data.

## Quick Start

**Terminal 1 - Backend:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
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

## Windows Troubleshooting

### "Character maps to undefined" Error

If you get an encoding error like `"character maps to undefined"` on Windows:

**Option 1: Use the fix script (Recommended)**
```bash
python fix_csv_encoding.py --all
```

**Option 2: Fix specific file**
```bash
python fix_csv_encoding.py your_data.csv
```

This converts CSV files to UTF-8 with BOM, which Windows handles reliably.

**What causes this?**
- CSV files created on Mac/Linux may use different character encodings
- Windows defaults to different encodings (CP1252) vs Unix (UTF-8)
- Special characters or byte order marks (BOM) can cause issues

All code now uses `encoding='utf-8-sig'` for cross-platform compatibility.

---

> **Python 3.14 Note**: Uses statistical fallback generator (CTGAN requires Python ≤3.13)
