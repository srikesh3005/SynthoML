#!/usr/bin/env python3
"""
Simple statistical synthetic data generator.
Fallback for Python 3.14+ where CTGAN is not available.
Uses pandas and numpy to generate data based on statistical distributions.
"""

import pandas as pd
import numpy as np
import joblib
from typing import Optional


class SimpleGenerator:
    """Statistical synthetic data generator using distribution sampling."""
    
    def __init__(self):
        self.columns = []
        self.categorical_columns = []
        self.stats = {}
        
    def fit(self, df: pd.DataFrame, categorical_columns: list = None):
        """Learn statistics from the training data."""
        self.columns = list(df.columns)
        self.categorical_columns = categorical_columns or []
        
        for col in df.columns:
            if col in self.categorical_columns or df[col].dtype == 'object':
                # For categorical: store value counts (probabilities)
                value_counts = df[col].value_counts(normalize=True)
                self.stats[col] = {
                    'type': 'categorical',
                    'values': list(value_counts.index),
                    'probabilities': list(value_counts.values)
                }
            else:
                # For numerical: store mean, std, min, max
                self.stats[col] = {
                    'type': 'numerical',
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max())
                }
    
    def sample(self, n: int, seed: Optional[int] = None) -> pd.DataFrame:
        """Generate n synthetic samples."""
        if seed is not None:
            np.random.seed(seed)
        
        synthetic_data = {}
        
        for col in self.columns:
            col_stats = self.stats[col]
            
            if col_stats['type'] == 'categorical':
                # Sample from categorical distribution
                synthetic_data[col] = np.random.choice(
                    col_stats['values'],
                    size=n,
                    p=col_stats['probabilities']
                )
            else:
                # Sample from normal distribution, clipped to min/max
                samples = np.random.normal(
                    col_stats['mean'],
                    col_stats['std'],
                    size=n
                )
                # Clip to observed range
                samples = np.clip(samples, col_stats['min'], col_stats['max'])
                synthetic_data[col] = samples
        
        return pd.DataFrame(synthetic_data)


def train_simple_generator(data_path: str, categorical_cols: list = None):
    """Train and return a simple generator model."""
    df = pd.read_csv(data_path)
    
    # Auto-detect categorical if not specified
    if categorical_cols is None:
        categorical_cols = []
        for col in df.columns:
            if df[col].dtype == 'object':
                categorical_cols.append(col)
            elif df[col].nunique() <= 20:  # Low cardinality
                categorical_cols.append(col)
    
    generator = SimpleGenerator()
    generator.fit(df, categorical_cols)
    
    return {
        'model': generator,
        'library': 'simple-statistical',
        'categorical_columns': categorical_cols,
        'columns': list(df.columns)
    }


def save_simple_model(model_data: dict, output_path: str = 'ctgan_model.joblib'):
    """Save the simple generator model."""
    joblib.dump(model_data, output_path)
    print(f"✓ Simple generator model saved to {output_path}")


if __name__ == '__main__':
    # Test the simple generator
    print("Testing Simple Statistical Generator...")
    
    # Create sample data
    test_data = pd.DataFrame({
        'age': [25, 30, 35, 40, 45, 50, 55, 60],
        'gender': ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'F'],
        'score': [65.5, 72.3, 68.9, 85.1, 90.2, 78.5, 82.3, 76.8]
    })
    
    print("\nOriginal data:")
    print(test_data)
    
    # Train generator
    model_data = train_simple_generator('toy_medical.csv')
    
    # Generate samples
    generator = model_data['model']
    synthetic = generator.sample(10, seed=42)
    
    print("\nGenerated synthetic data:")
    print(synthetic)
    
    print("\nStatistics comparison:")
    print("Original mean:", test_data.select_dtypes(include=[np.number]).mean().to_dict())
    print("Synthetic mean:", synthetic.select_dtypes(include=[np.number]).mean().to_dict())
