#!/usr/bin/env python3

import argparse
import sys
from inference import generate, get_model_info


def main():
    parser = argparse.ArgumentParser(
        description='Test CTGAN inference by generating synthetic data'
    )
    parser.add_argument(
        '--n',
        type=int,
        default=100,
        help='Number of synthetic samples to generate (default: 100)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (optional)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='test_synthetic_output.csv',
        help='Output CSV file path (default: test_synthetic_output.csv)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='ctgan_model.joblib',
        help='Path to model file (default: ctgan_model.joblib)'
    )
    
    args = parser.parse_args()
    
    print("\nCTGAN Inference Test\n")
    
    try:
        print("Model Information:")
        info = get_model_info(args.model)
        print(f"   Library: {info['library']}")
        print(f"   Columns: {', '.join(info['columns'])}")
        print(f"   Categorical: {', '.join(info['categorical_columns'])}")
        
        print(f"\nGenerating {args.n} synthetic samples...")
        if args.seed is not None:
            print(f"   Using random seed: {args.seed}")
        
        synthetic_df = generate(n=args.n, seed=args.seed, model_path=args.model)
        
        print(f"\nFirst 10 rows:")
        print(synthetic_df.head(10))
        
        print(f"\nSummary Statistics:")
        print(synthetic_df.describe())
        
        synthetic_df.to_csv(args.output, index=False)
        print(f"\nSaved {len(synthetic_df)} rows to {args.output}")
        
        print("\nTest completed successfully!")
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease run the training script first:")
        print("   python train_and_save_ctgan.py --data toy_medical.csv")
        sys.exit(1)
        
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
