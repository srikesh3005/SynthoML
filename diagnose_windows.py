#!/usr/bin/env python3
"""
Diagnose encoding and data issues for Windows.
Run this to identify exactly what's causing the training error.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path


def check_csv_encoding(file_path):
    """Check if CSV can be read with various encodings."""
    print(f"\n{'='*60}")
    print(f"Checking: {file_path}")
    print('='*60)
    
    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        return False
    
    encodings = ['utf-8-sig', 'utf-8', 'latin1', 'cp1252', 'iso-8859-1', 'ascii']
    working_encoding = None
    df = None
    
    print("\n1. Testing encodings:")
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            print(f"   ✓ {enc:15} - SUCCESS ({len(df)} rows, {len(df.columns)} cols)")
            if working_encoding is None:
                working_encoding = enc
        except Exception as e:
            print(f"   ✗ {enc:15} - FAILED: {str(e)[:50]}")
    
    if df is None:
        print("\n❌ Could not read file with any encoding!")
        return False
    
    print(f"\n2. File Info:")
    print(f"   - Best encoding: {working_encoding}")
    print(f"   - Shape: {df.shape}")
    print(f"   - Columns: {list(df.columns)}")
    
    print(f"\n3. Checking for problematic characters:")
    issues_found = False
    
    for col in df.columns:
        # Check column name
        try:
            col.encode('ascii')
        except UnicodeEncodeError as e:
            print(f"   ⚠️  Column name '{col}' has non-ASCII: {e}")
            issues_found = True
        
        # Check data values for string columns
        if df[col].dtype == 'object':
            problematic_values = []
            for idx, val in df[col].items():
                if pd.notna(val):
                    try:
                        str(val).encode('ascii')
                    except UnicodeEncodeError:
                        problematic_values.append((idx, val))
                        if len(problematic_values) <= 3:  # Show first 3 examples
                            print(f"   ⚠️  Col '{col}', Row {idx}: Non-ASCII character in '{val}'")
                            issues_found = True
            
            if len(problematic_values) > 3:
                print(f"   ⚠️  Col '{col}': {len(problematic_values)} more rows with non-ASCII chars")
    
    if not issues_found:
        print("   ✓ No non-ASCII characters found")
    
    print(f"\n4. Data types:")
    for col in df.columns:
        unique_count = df[col].nunique()
        null_count = df[col].isna().sum()
        print(f"   - {col:20} | {str(df[col].dtype):10} | {unique_count:4} unique | {null_count:3} nulls")
    
    print(f"\n5. Sample data (first 3 rows):")
    print(df.head(3).to_string())
    
    return True


def check_python_env():
    """Check Python environment."""
    print(f"\n{'='*60}")
    print("Python Environment")
    print('='*60)
    print(f"Python version: {sys.version}")
    print(f"Platform: {sys.platform}")
    print(f"Default encoding: {sys.getdefaultencoding()}")
    
    try:
        import locale
        print(f"Locale encoding: {locale.getpreferredencoding()}")
    except:
        pass
    
    print(f"\nInstalled packages:")
    for pkg in ['pandas', 'numpy', 'joblib', 'sdv', 'ctgan']:
        try:
            mod = __import__(pkg)
            version = getattr(mod, '__version__', 'unknown')
            print(f"   ✓ {pkg:15} {version}")
        except ImportError:
            print(f"   ✗ {pkg:15} NOT INSTALLED")


def main():
    print("="*60)
    print("WINDOWS ENCODING DIAGNOSTIC TOOL")
    print("="*60)
    
    check_python_env()
    
    # Check common files
    files_to_check = [
        'uploaded_data.csv',
        'toy_medical.csv',
        'sample_synthetic_preview.csv'
    ]
    
    for file in files_to_check:
        if Path(file).exists():
            check_csv_encoding(file)
    
    # Check for custom file
    if len(sys.argv) > 1:
        check_csv_encoding(sys.argv[1])
    
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS:")
    print('='*60)
    print("1. Run: python fix_csv_encoding.py --all")
    print("2. Then try training again")
    print("3. If still fails, check that all data is ASCII-compatible")
    print("4. Windows locale/encoding issues may require ASCII-only data")


if __name__ == '__main__':
    main()
