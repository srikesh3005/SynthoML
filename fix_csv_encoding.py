#!/usr/bin/env python3
"""
Fix CSV encoding issues for Windows compatibility.
This script converts CSV files to UTF-8 with BOM, which works reliably on Windows.
"""

import sys
import argparse
import pandas as pd
from pathlib import Path


def fix_csv_encoding(input_path: str, output_path: str = None):
    """
    Read a CSV file and re-save it with UTF-8-sig encoding (UTF-8 with BOM).
    
    Args:
        input_path: Path to the input CSV file
        output_path: Path to save the fixed CSV (defaults to overwriting input)
    """
    if output_path is None:
        output_path = input_path
    
    print(f"Reading {input_path}...")
    
    # Try different encodings
    encodings_to_try = ['utf-8-sig', 'utf-8', 'latin1', 'cp1252', 'iso-8859-1']
    df = None
    encoding_used = None
    
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(input_path, encoding=encoding)
            encoding_used = encoding
            print(f"  [OK] Successfully read with encoding: {encoding}")
            break
        except (UnicodeDecodeError, UnicodeError) as e:
            continue
        except Exception as e:
            print(f"  Error with {encoding}: {e}")
            continue
    
    if df is None:
        print(f"  ✗ Failed to read file with any common encoding")
        return False
    
    print(f"  Loaded: {len(df)} rows, {len(df.columns)} columns")
    
    # Clean string columns for Windows compatibility
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('').astype(str)
            # Convert to ASCII, removing non-ASCII characters
            df[col] = df[col].apply(lambda x: x.encode('ascii', 'ignore').decode('ascii') if isinstance(x, str) else str(x))
            df[col] = df[col].str.strip()
            df[col] = df[col].replace('', 'Unknown')
    
    # Save with UTF-8-sig (includes BOM for Windows compatibility)
    print(f"Writing to {output_path} with UTF-8-sig encoding...")
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"  [OK] File saved successfully")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Fix CSV encoding for Windows compatibility'
    )
    parser.add_argument(
        'input',
        type=str,
        nargs='?',
        help='Path to input CSV file (if not provided, fixes common files)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Path to output CSV file (default: overwrites input)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Fix all CSV files in the current directory'
    )
    
    args = parser.parse_args()
    
    if args.all:
        # Fix all CSV files in current directory
        csv_files = list(Path('.').glob('*.csv'))
        if not csv_files:
            print("No CSV files found in current directory")
            return
        
        print(f"Found {len(csv_files)} CSV files to process\n")
        success_count = 0
        
        for csv_file in csv_files:
            print(f"\n{'='*60}")
            if fix_csv_encoding(str(csv_file)):
                success_count += 1
        
        print(f"\n{'='*60}")
        print(f"Processed {success_count}/{len(csv_files)} files successfully")
        
    elif args.input:
        # Fix specific file
        fix_csv_encoding(args.input, args.output)
        
    else:
        # Fix common files
        common_files = ['toy_medical.csv', 'uploaded_data.csv']
        print("Fixing common CSV files...\n")
        
        for file in common_files:
            if Path(file).exists():
                print(f"{'='*60}")
                fix_csv_encoding(file)
                print()
            else:
                print(f"Skipping {file} (not found)")
                print()


if __name__ == '__main__':
    main()
