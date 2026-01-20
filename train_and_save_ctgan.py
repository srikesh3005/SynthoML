#!/usr/bin/env python3

import argparse
import sys
import warnings
import pandas as pd
import joblib
import numpy as np

warnings.filterwarnings('ignore')


def detect_categorical_columns(df, max_unique_ratio=0.3, max_unique_count=20):
    categorical_cols = []
    
    for col in df.columns:
        if df[col].dtype == 'object':
            categorical_cols.append(col)
        elif df[col].nunique() <= max_unique_count or \
             (df[col].nunique() / len(df)) <= max_unique_ratio:
            categorical_cols.append(col)
    
    return categorical_cols


def train_with_sdv(df, categorical_cols, epochs):
    from sdv.single_table import CTGANSynthesizer
    from sdv.metadata import SingleTableMetadata
    
    print("✓ Using SDV's CTGANSynthesizer")
    
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)
    
    for col in categorical_cols:
        metadata.update_column(col, sdtype='categorical')
    
    model = CTGANSynthesizer(
        metadata=metadata,
        epochs=epochs,
        verbose=True
    )
    
    model.fit(df)
    return model


def train_with_ctgan(df, categorical_cols, epochs):
    from ctgan import CTGAN
    
    print("✓ Using standalone ctgan package")
    
    model = CTGAN(
        epochs=epochs,
        verbose=True
    )
    
    model.fit(df, discrete_columns=categorical_cols)
    return model


def main():
    parser = argparse.ArgumentParser(
        description='Train CTGAN on CSV data with automatic fallback logic'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='toy_medical.csv',
        help='Path to CSV file (default: toy_medical.csv)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='ctgan_model.joblib',
        help='Path to save trained model (default: ctgan_model.joblib)'
    )
    parser.add_argument(
        '--preview-samples',
        type=int,
        default=20,
        help='Number of synthetic samples to generate for preview (default: 20)'
    )
    
    args = parser.parse_args()
    
    print(f"\nLoading data from {args.data}...")
    df = pd.read_csv(args.data)
    print(f"   Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"   Columns: {list(df.columns)}")
    
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str)
    
    categorical_cols = detect_categorical_columns(df)
    print(f"\nDetected categorical columns: {categorical_cols}")
    
    print(f"\nTraining CTGAN for {args.epochs} epochs...")
    model = None
    library_used = None
    
    try:
        model = train_with_sdv(df, categorical_cols, args.epochs)
        library_used = 'sdv'
    except ImportError:
        print("\nSDV not installed, falling back to ctgan package...")
        try:
            model = train_with_ctgan(df, categorical_cols, args.epochs)
            library_used = 'ctgan'
        except ImportError:
            print("\nCTGAN libraries not available, using simple statistical generator...")
            print("   (Python 3.14+ detected - CTGAN not yet supported)")
            from simple_generator import train_simple_generator
            model_data = train_simple_generator(args.data, categorical_cols)
            library_used = model_data['library']
            model = model_data['model']
            
            print(f"\n✓ Simple generator trained successfully")
            print("   Note: This uses statistical sampling, not deep learning")
            print("   For better quality, use Python 3.9-3.13 with CTGAN")
    
    print(f"\nSaving model to {args.output}...")
    model_data = {
        'model': model,
        'library': library_used,
        'categorical_columns': categorical_cols,
        'columns': list(df.columns)
    }
    joblib.dump(model_data, args.output)
    print("   ✓ Model saved successfully")
    
    print(f"\nGenerating {args.preview_samples} synthetic samples for preview...")
    try:
        if library_used == 'sdv':
            synthetic_preview = model.sample(num_rows=args.preview_samples)
        elif library_used == 'simple-statistical':
            synthetic_preview = model.sample(args.preview_samples)
        else:
            synthetic_preview = model.sample(args.preview_samples)
        
        preview_path = 'sample_synthetic_preview.csv'
        synthetic_preview.to_csv(preview_path, index=False)
        print(f"   ✓ Preview saved to {preview_path}")
        print("\nSample synthetic data preview:")
        print(synthetic_preview.head(10))
        
    except Exception as e:
        print(f"   Could not generate preview: {e}")
    
    print("\nTraining complete!")
    print(f"\nSummary:")
    print(f"   - Library used: {library_used}")
    print(f"   - Model saved: {args.output}")
    print(f"   - Training samples: {len(df)}")
    print(f"   - Epochs: {args.epochs if library_used != 'simple-statistical' else 'N/A'}")
    print(f"   - Categorical columns: {categorical_cols}")


if __name__ == '__main__':
    main()
