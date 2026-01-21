# Windows Encoding Fix - Complete Guide

## Problem
Error: `"training failed traceback in position 2: character maps to undefined"`

## Root Cause
Windows has different default encodings (CP1252/MBCS) compared to Unix systems (UTF-8). Non-ASCII characters in CSV data cause encoding errors during training or model serialization.

## Complete Solution Applied

### 1. ASCII-Only Data Processing
**All string data is now forced to ASCII:**
- Non-ASCII characters are stripped: `x.encode('ascii', 'ignore').decode('ascii')`
- Empty strings replaced with "Unknown"
- Applied in:
  - ✅ `train_and_save_ctgan.py` - Training script
  - ✅ `simple_generator.py` - Fallback generator  
  - ✅ `server.py` - Web upload processing
  - ✅ `fix_csv_encoding.py` - CSV repair tool

### 2. Pickle Protocol 4
**Model serialization now uses protocol 4:**
- Better cross-platform compatibility
- Prevents encoding issues in joblib.dump()
- Works reliably on Windows, Mac, Linux

### 3. UTF-8-sig Encoding
**All CSV operations use UTF-8 with BOM:**
- `pd.read_csv(..., encoding='utf-8-sig')`
- `df.to_csv(..., encoding='utf-8-sig')`
- Windows-friendly with proper BOM handling

## On Your Windows Machine

### Step 1: Pull Latest Code
```bash
git pull
```

### Step 2: Diagnose Issues (Optional)
```bash
# See exactly what's wrong with your data
python diagnose_windows.py
```

### Step 3: Fix Existing CSV Files
```bash
# Fix all CSV files - strips non-ASCII characters
python fix_csv_encoding.py --all
```

### Step 4: Restart Backend Server
```bash
# Stop current server (Ctrl+C), then:
python server.py
```

### Step 5: Try Training
**Option A - Via Web Interface:**
1. Upload CSV through browser
2. Server automatically cleans data
3. Train model

**Option B - Command Line:**
```bash
python train_and_save_ctgan.py --data uploaded_data.csv --epochs 100
```

## What Changed

### Before:
- Raw UTF-8 encoding could have problematic characters
- No ASCII normalization
- Default pickle protocol (5)
- Characters like é, ñ, ™, © caused errors

### After:
- All strings forced to ASCII
- Non-ASCII chars automatically removed
- Pickle protocol 4 for compatibility
- Only printable ASCII characters allowed

## Diagnostic Tool

Run `diagnose_windows.py` to check:
- ✓ Which encodings work with your file
- ✓ Exact location of problematic characters
- ✓ Data type information
- ✓ Python environment details

```bash
python diagnose_windows.py
python diagnose_windows.py your_file.csv
```

## Still Having Issues?

If the error persists:

1. **Check the exact error message** - Note the position number
2. **Run diagnostics**:
   ```bash
   python diagnose_windows.py uploaded_data.csv
   ```
3. **Check your data** - Ensure no binary/corrupted data mixed in
4. **Try with toy data**:
   ```bash
   python train_and_save_ctgan.py --data toy_medical.csv --epochs 10
   ```

## Technical Details

### The Error
`"character maps to undefined"` occurs when:
- Python tries to encode a string with a codec that doesn't support certain characters
- Joblib/pickle tries to serialize objects with problematic strings
- Windows locale settings conflict with data encoding

### The Fix
1. **Strip all non-ASCII**: `encode('ascii', 'ignore')` removes problematic chars
2. **Normalize encoding**: UTF-8-sig provides explicit BOM for Windows
3. **Protocol 4**: More stable pickle format across platforms
4. **Defensive coding**: Try multiple encodings, handle edge cases
