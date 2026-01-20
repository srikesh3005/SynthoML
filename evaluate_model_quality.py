#!/usr/bin/env python3
"""
Evaluate CTGAN model quality by comparing synthetic vs real data.
Reports statistical similarity metrics and correlation preservation.
"""

import argparse
import pandas as pd
import numpy as np
from inference import generate
import joblib


def calculate_column_statistics(df, col):
    """Calculate statistics for a column."""
    stats = {}
    if pd.api.types.is_numeric_dtype(df[col]):
        stats['mean'] = df[col].mean()
        stats['std'] = df[col].std()
        stats['min'] = df[col].min()
        stats['max'] = df[col].max()
        stats['median'] = df[col].median()
    else:
        # Categorical
        value_counts = df[col].value_counts(normalize=True)
        stats['distribution'] = value_counts.to_dict()
    return stats


def compare_distributions(real_df, synthetic_df):
    """Compare distributions between real and synthetic data."""
    print("\n" + "="*80)
    print("DISTRIBUTION COMPARISON")
    print("="*80)
    
    scores = []
    
    for col in real_df.columns:
        print(f"\n{col}:")
        
        if pd.api.types.is_numeric_dtype(real_df[col]):
            real_mean = real_df[col].mean()
            syn_mean = synthetic_df[col].mean()
            real_std = real_df[col].std()
            syn_std = synthetic_df[col].std()
            
            mean_diff = abs(real_mean - syn_mean) / (abs(real_mean) + 1e-10)
            std_diff = abs(real_std - syn_std) / (abs(real_std) + 1e-10)
            
            print(f"  Mean:   Real={real_mean:.4f}, Synthetic={syn_mean:.4f}, Diff={mean_diff:.4%}")
            print(f"  Std:    Real={real_std:.4f}, Synthetic={syn_std:.4f}, Diff={std_diff:.4%}")
            
            # Score: higher is better (max 1.0)
            score = 1.0 - min(1.0, (mean_diff + std_diff) / 2)
            scores.append(score)
            print(f"  Similarity Score: {score:.2%}")
            
        else:
            # Categorical
            real_dist = real_df[col].value_counts(normalize=True)
            syn_dist = synthetic_df[col].value_counts(normalize=True)
            
            # Calculate Total Variation Distance
            all_categories = set(real_dist.index) | set(syn_dist.index)
            tvd = sum(abs(real_dist.get(cat, 0) - syn_dist.get(cat, 0)) 
                     for cat in all_categories) / 2
            
            print(f"  Total Variation Distance: {tvd:.4f}")
            print(f"  Real categories: {len(real_dist)}, Synthetic: {len(syn_dist)}")
            
            # Show top categories
            print(f"  Top 3 Real: {dict(list(real_dist.head(3).items()))}")
            print(f"  Top 3 Syn:  {dict(list(syn_dist.head(3).items()))}")
            
            score = 1.0 - tvd
            scores.append(score)
            print(f"  Similarity Score: {score:.2%}")
    
    return scores


def compare_correlations(real_df, synthetic_df):
    """Compare correlation matrices between real and synthetic data."""
    print("\n" + "="*80)
    print("CORRELATION PRESERVATION")
    print("="*80)
    
    # Only numeric columns
    numeric_cols = real_df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) < 2:
        print("  Not enough numeric columns to compute correlations")
        return None
    
    real_corr = real_df[numeric_cols].corr()
    syn_corr = synthetic_df[numeric_cols].corr()
    
    # Flatten and compare
    real_corr_flat = real_corr.values[np.triu_indices_from(real_corr.values, k=1)]
    syn_corr_flat = syn_corr.values[np.triu_indices_from(syn_corr.values, k=1)]
    
    # Calculate correlation of correlations
    mae = np.mean(np.abs(real_corr_flat - syn_corr_flat))
    
    # Handle case with insufficient data
    if len(real_corr_flat) > 0:
        corr_of_corr = np.corrcoef(real_corr_flat, syn_corr_flat)[0, 1]
        if np.isnan(corr_of_corr):
            # Use MAE-based score if correlation fails
            corr_of_corr = 1.0 - mae
    else:
        corr_of_corr = 1.0 - mae
    
    print(f"\n  Correlation of Correlations: {corr_of_corr:.4f}")
    print(f"  Mean Absolute Error: {mae:.4f}")
    print(f"  Correlation Preservation Score: {corr_of_corr:.2%}")
    
    # Show some examples
    print(f"\n  Example correlations:")
    for i in range(min(3, len(numeric_cols)-1)):
        for j in range(i+1, min(i+2, len(numeric_cols))):
            col1, col2 = numeric_cols[i], numeric_cols[j]
            print(f"    {col1} <-> {col2}:")
            print(f"      Real: {real_corr.loc[col1, col2]:.4f}, Synthetic: {syn_corr.loc[col1, col2]:.4f}")
    
    return corr_of_corr


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate CTGAN model quality'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='toy_medical.csv',
        help='Path to original CSV file (default: toy_medical.csv)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='ctgan_model.joblib',
        help='Path to model file (default: ctgan_model.joblib)'
    )
    parser.add_argument(
        '--n',
        type=int,
        default=None,
        help='Number of synthetic samples to generate (default: same as real data)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("CTGAN MODEL QUALITY EVALUATION")
    print("="*80)
    
    # Load real data
    print(f"\nLoading real data from {args.data}...")
    real_df = pd.read_csv(args.data)
    print(f"  Shape: {real_df.shape}")
    
    # Load model info
    model_data = joblib.load(args.model)
    print(f"\nModel Information:")
    print(f"  Library: {model_data['library']}")
    print(f"  Columns: {len(model_data['columns'])}")
    
    # Generate synthetic data
    n_samples = args.n if args.n is not None else len(real_df)
    print(f"\nGenerating {n_samples} synthetic samples...")
    synthetic_df = generate(n=n_samples, seed=args.seed, model_path=args.model)
    print(f"  Shape: {synthetic_df.shape}")
    
    # Compare distributions
    dist_scores = compare_distributions(real_df, synthetic_df)
    
    # Compare correlations
    corr_score = compare_correlations(real_df, synthetic_df)
    
    # Overall score
    print("\n" + "="*80)
    print("OVERALL QUALITY METRICS")
    print("="*80)
    
    avg_dist_score = np.mean(dist_scores)
    print(f"\nAverage Distribution Similarity: {avg_dist_score:.2%}")
    
    if corr_score is not None and not np.isnan(corr_score):
        print(f"Correlation Preservation:        {corr_score:.2%}")
        overall_score = (avg_dist_score + corr_score) / 2
        print(f"\nOVERALL QUALITY SCORE:           {overall_score:.2%}")
    else:
        overall_score = avg_dist_score
        print(f"\nOVERALL QUALITY SCORE:           {overall_score:.2%}")
    
    print("\nInterpretation:")
    if overall_score >= 0.85:
        print("  ✓ Excellent - Synthetic data closely matches real data")
    elif overall_score >= 0.70:
        print("  ✓ Good - Synthetic data captures most patterns")
    elif overall_score >= 0.50:
        print("  ~ Fair - Some patterns preserved, room for improvement")
    else:
        print("  ✗ Poor - Consider retraining with more epochs")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
