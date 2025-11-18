"""
Analyze training results and find best configurations for each model type
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path

# Default values
DEFAULT_TASK = "regression"
MODEL_FOLDER = "./saved_models/"
DIAGNOSTICS_FOLDER = "./diagnostics/"
OUTPUT_FILE = DIAGNOSTICS_FOLDER + "best_configs.csv"

def collect_all_results() -> pd.DataFrame:
    """
    Scan all model directories and collect specifications
    
    Returns:
        DataFrame with all model specifications and performance metrics
    """
    all_results = []
    
    # Iterate through all model directories
    for model_dir in Path(MODEL_FOLDER).iterdir():
        if not model_dir.is_dir():
            continue
        
        specs_file = model_dir / "specs.csv"
        if not specs_file.exists():
            continue
        
        # Read specs file
        specs_df = pd.read_csv(specs_file, index_col=0)
        specs_dict = specs_df['value'].to_dict()
        
        # Extract relevant information
        result = {
            'model_name': specs_dict['model_name'],
            'model_type': specs_dict['model_type'],
            'n_parameters': int(specs_dict['n_parameters']),
            'val_loss': float(specs_dict['val_loss']),
            'val_metric': float(specs_dict['val_metric'])
        }
        
        # Add model-specific hyperparameters
        for key, value in specs_dict.items():
            if key not in result and key != 'model_architecture':
                try:
                    # Try to convert to numeric if possible
                    result[key] = float(value) if '.' in str(value) else int(value)
                except (ValueError, TypeError):
                    result[key] = value
        
        all_results.append(result)

    return pd.DataFrame(all_results)

def create_visualizations(df: pd.DataFrame, best_configs: pd.DataFrame) -> None:
    """
    Create comparison visualizations
    
    Args:
        df: DataFrame with all results
        best_configs: DataFrame with best configurations
    """
    # Create output directory for plots
    plot_dir = Path(DIAGNOSTICS_FOLDER) / "analysis_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Box plot of validation losses by model type
    plt.figure(figsize=(12, 6))
    df.boxplot(column='val_metric', by='model_type', figsize=(12, 6))
    plt.title('Validation Metric Distribution by Model Type')
    plt.suptitle('')
    plt.xlabel('Model Type')
    plt.ylabel('Validation Metric')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(plot_dir / 'val_metric_by_model_type.png', dpi=300)
    plt.close()
    
    # 2. Bar plot of best configurations
    plt.figure(figsize=(10, 6))
    plt.barh(best_configs['model_type'], best_configs['val_metric'])
    plt.xlabel('Best Validation Metric')
    plt.ylabel('Model Type')
    plt.title('Validation Metric for Each Model Type')
    plt.tight_layout()
    plt.savefig(plot_dir / 'best_configs_comparison.png', dpi=300)
    plt.close()
    
    # 3. Scatter plot: number of parameters vs validation loss
    plt.figure(figsize=(10, 6))
    for model_type in df['model_type'].unique():
        subset = df[df['model_type'] == model_type]
        plt.scatter(subset['n_parameters'], subset['val_metric'], 
                   label=model_type, alpha=0.6, s=50)
    
    plt.xlabel('Number of Parameters')
    plt.ylabel('Best Validation Metric')
    plt.title('Model Complexity vs Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / 'params_vs_loss.png', dpi=300)
    plt.close()
    
    print(f"\nVisualizations saved to {plot_dir}/")

def main() -> None:
    """
    Main function to analyze all results
    
    Parses command-line arguments, collects results from trained models,
    finds best configurations, creates visualizations, and saves outputs.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze training results')
    parser.add_argument('--task', type=str, default='regression', 
                        choices=['regression', 'classification'],
                        help='Task type: regression or classification')
    args = parser.parse_args()
    
    task = args.task
    
    # Update paths based on task
    global MODEL_FOLDER, DIAGNOSTICS_FOLDER, OUTPUT_FILE
    MODEL_FOLDER = f"./saved_models/{task}/"
    DIAGNOSTICS_FOLDER = f"./diagnostics/{task}/"
    OUTPUT_FILE = DIAGNOSTICS_FOLDER + "best_configs.csv"
    
    print(f"Task: {task}")
    print(f"Collecting results from all trained models in {MODEL_FOLDER}...")
    df = collect_all_results()
    
    print(f"Found {len(df)} trained models.")
    
    # Find best configurations
    best_configs = df.loc[df.groupby('model_type')['val_metric'].idxmin()]
    
    # Create diagnostics folder if it doesn't exist
    os.makedirs(DIAGNOSTICS_FOLDER, exist_ok=True)
    
    # Save best configurations
    best_configs.to_csv(OUTPUT_FILE, index=False)
    print(f"\nBest configurations saved to: {OUTPUT_FILE}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(df, best_configs)
    
    # Save full results too
    full_results_file = Path(DIAGNOSTICS_FOLDER) / "all_results.csv"
    df.to_csv(full_results_file, index=False)
    print(f"Full results saved to: {full_results_file}")

if __name__ == "__main__":
    main()
