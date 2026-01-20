# Windows Encoding Fix - Quick Guide

## Problem
Error: `"training failed traceback in position 2: character maps to undefined"`

## Root Cause
CSV files created on Mac/Linux may have encoding issues on Windows because:
- Different default encodings (UTF-8 on Unix vs CP1252 on Windows)
- Byte Order Mark (BOM) differences
- Special characters not handled properly

## Solution Applied

### 1. Code Changes
All Python files now use `encoding='utf-8-sig'` when reading CSV files:
- ✅ `train_and_save_ctgan.py` - Main training script
- ✅ `simple_generator.py` - Fallback generator
- ✅ `evaluate_model_quality.py` - Evaluation script
- ✅ `server.py` - Web server

### 2. Fix Script Created
Use `fix_csv_encoding.py` to convert existing CSV files:

```bash
# Fix all CSV files in current directory
python fix_csv_encoding.py --all

# Fix specific file
python fix_csv_encoding.py your_data.csv

# Fix and save to different file
python fix_csv_encoding.py input.csv --output output.csv
```

## On Your Windows Machine

### Step 1: Pull Latest Code
```bash
git pull  # or download the updated files
```

### Step 2: Fix Your Data Files
```bash
python fix_csv_encoding.py --all
```

### Step 3: Try Training Again
```bash
python train_and_save_ctgan.py --data toy_medical.csv --epochs 100
```

## Testing
The script automatically tries multiple encodings:
1. utf-8-sig (UTF-8 with BOM - best for Windows)
2. utf-8 (Standard UTF-8)
3. latin1 (ISO-8859-1)
4. cp1252 (Windows default)
5. iso-8859-1

Then saves everything as UTF-8 with BOM for maximum compatibility.

## If Issue Persists
Check the specific error message and verify:
1. CSV file isn't corrupted
2. File has proper line endings (CRLF on Windows is fine)
3. No binary data mixed in CSV file

You can also check the file encoding with:
```bash
file your_data.csv
```
