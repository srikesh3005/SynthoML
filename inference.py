#!/usr/bin/env python3

import joblib
import pandas as pd
from typing import Optional

_model_cache = None


def load_model(model_path: str = "ctgan_model.joblib") -> dict:
    global _model_cache
    
    if _model_cache is None:
        print(f"Loading model from {model_path}...")
        try:
            _model_cache = joblib.load(model_path)
            print(f"✓ Model loaded successfully (library: {_model_cache['library']})")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Model file '{model_path}' not found. "
                "Please run train_and_save_ctgan.py first."
            )
        except Exception as e:
            raise Exception(f"Failed to load model: {e}")
    
    return _model_cache


def generate(n: int = 1000, seed: Optional[int] = None, model_path: str = "ctgan_model.joblib") -> pd.DataFrame:
    if n <= 0:
        raise ValueError(f"Number of samples must be positive, got {n}")
    
    model_data = load_model(model_path)
    model = model_data['model']
    library = model_data['library']
    
    print(f"Generating {n} synthetic samples...")
    
    try:
        if library == 'sdv':
            try:
                synthetic_df = model.sample(num_rows=n, random_state=seed)
            except TypeError:
                if seed is not None:
                    print(f"Warning: random_state not supported by this SDV version")
                synthetic_df = model.sample(num_rows=n)
        
        elif library == 'simple-statistical':
            synthetic_df = model.sample(n, seed=seed)
        
        else:  # ctgan
            if seed is not None:
                import numpy as np
                np.random.seed(seed)
            synthetic_df = model.sample(n)
        
        print(f"✓ Generated {len(synthetic_df)} rows")
        return synthetic_df
        
    except Exception as e:
        raise Exception(f"Failed to generate synthetic data: {e}")


def get_model_info(model_path: str = "ctgan_model.joblib") -> dict:
    model_data = load_model(model_path)
    
    return {
        'library': model_data.get('library', 'unknown'),
        'columns': model_data.get('columns', []),
        'categorical_columns': model_data.get('categorical_columns', [])
    }


if __name__ == '__main__':
    print("Testing inference module...")
    
    try:
        info = get_model_info()
        print(f"\nModel info:")
        print(f"  Library: {info['library']}")
        print(f"  Columns: {info['columns']}")
        print(f"  Categorical: {info['categorical_columns']}")
        
        df = generate(n=5, seed=42)
        print(f"\nGenerated sample:")
        print(df)
        
    except FileNotFoundError as e:
        print(f"\n{e}")
        print("Please run train_and_save_ctgan.py first.")
    except Exception as e:
        print(f"\nError: {e}")
