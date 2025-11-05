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
        
        try:
            # Read specs file
            specs_df = pd.read_csv(specs_file, index_col=0)
            specs_dict = specs_df['value'].to_dict()
            
            # Extract relevant information
            result = {
                'model_name': specs_dict.get('model_name', model_dir.name),
                'model_type': specs_dict.get('model_type', 'unknown'),
                'n_parameters': int(specs_dict.get('n_parameters', 0)),
                'best_val_loss': float(specs_dict.get('best_val_loss', specs_dict.get('mean_val_loss', float('inf')))),
                'mean_val_loss': float(specs_dict.get('mean_val_loss', specs_dict.get('best_val_loss', float('inf')))),
                'std_val_loss': float(specs_dict.get('std_val_loss', 0)),
                'learning_rate': float(specs_dict.get('learning_rate', 0)),
                'weight_decay': float(specs_dict.get('weight_decay', 0)),
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
        
        except Exception as e:
            print(f"Error processing {model_dir.name}: {e}")
            continue
    
    return pd.DataFrame(all_results)

def find_best_configs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Find the best configuration for each model type (lowest mean validation loss)
    
    Args:
        df: DataFrame with all results
    
    Returns:
        DataFrame with best configuration for each model type, sorted by performance
    """
    # Group by model type and find the one with minimum mean validation loss
    # Use mean_val_loss if available, otherwise fall back to best_val_loss
    loss_column = 'mean_val_loss' if 'mean_val_loss' in df.columns else 'best_val_loss'
    best_configs = df.loc[df.groupby('model_type')[loss_column].idxmin()]
    
    # Sort by validation loss
    best_configs = best_configs.sort_values(loss_column)
    
    return best_configs

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
    df.boxplot(column='best_val_loss', by='model_type', figsize=(12, 6))
    plt.title('Validation Loss Distribution by Model Type')
    plt.suptitle('')
    plt.xlabel('Model Type')
    plt.ylabel('Best Validation Loss')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(plot_dir / 'val_loss_by_model_type.png', dpi=300)
    plt.close()
    
    # 2. Bar plot of best configurations
    plt.figure(figsize=(10, 6))
    plt.barh(best_configs['model_type'], best_configs['best_val_loss'])
    plt.xlabel('Best Validation Loss')
    plt.ylabel('Model Type')
    plt.title('Best Validation Loss for Each Model Type')
    plt.tight_layout()
    plt.savefig(plot_dir / 'best_configs_comparison.png', dpi=300)
    plt.close()
    
    # 3. Scatter plot: number of parameters vs validation loss
    plt.figure(figsize=(10, 6))
    for model_type in df['model_type'].unique():
        subset = df[df['model_type'] == model_type]
        plt.scatter(subset['n_parameters'], subset['best_val_loss'], 
                   label=model_type, alpha=0.6, s=50)
    
    plt.xlabel('Number of Parameters')
    plt.ylabel('Best Validation Loss')
    plt.title('Model Complexity vs Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_dir / 'params_vs_loss.png', dpi=300)
    plt.close()
    
    # 4. Heatmap of hyperparameter impact (if enough data)
    for model_type in df['model_type'].unique():
        subset = df[df['model_type'] == model_type]
        if len(subset) < 4:
            continue
        
        # Create pivot table for learning rate vs weight decay
        if 'learning_rate' in subset.columns and 'weight_decay' in subset.columns:
            pivot = subset.pivot_table(
                values='best_val_loss',
                index='learning_rate',
                columns='weight_decay',
                aggfunc='min'
            )
            
            if not pivot.empty:
                plt.figure(figsize=(10, 6))
                sns.heatmap(pivot, annot=True, fmt='.4f', cmap='viridis')
                plt.title(f'{model_type}: Learning Rate vs Weight Decay Impact')
                plt.tight_layout()
                plt.savefig(plot_dir / f'{model_type}_lr_wd_heatmap.png', dpi=300)
                plt.close()
    
    print(f"\nVisualizations saved to {plot_dir}/")

def print_summary(df, best_configs):
    """
    Print summary statistics
    
    Args:
        df (pd.DataFrame): DataFrame with all results
        best_configs (pd.DataFrame): DataFrame with best configurations
    """
    print("\n" + "="*80)
    print("TRAINING RESULTS SUMMARY")
    print("="*80)
    
    print(f"\nTotal models trained: {len(df)}")
    print(f"Model types: {df['model_type'].nunique()}")
    print(f"Model types list: {', '.join(df['model_type'].unique())}")
    
    print("\n" + "-"*80)
    print("BEST CONFIGURATION FOR EACH MODEL TYPE")
    print("-"*80)
    
    # Display best configs with selected columns
    display_cols = ['model_type', 'model_name', 'best_val_loss', 'n_parameters', 
                    'learning_rate', 'weight_decay']
    
    # Add model-specific hyperparameters if they exist
    for col in best_configs.columns:
        if col not in display_cols and col not in ['model_architecture']:
            display_cols.append(col)
    
    # Filter to only existing columns
    display_cols = [col for col in display_cols if col in best_configs.columns]
    
    print(best_configs[display_cols].to_string(index=False))
    
    print("\n" + "-"*80)


def print_summary(df: pd.DataFrame, best_configs: pd.DataFrame) -> None:
    """
    Print summary of all results and best configurations
    
    Args:
        df: DataFrame with all training results
        best_configs: DataFrame with best configuration per model type
    """
    print("OVERALL STATISTICS")
    print("-"*80)
    print(f"Best overall model: {best_configs.iloc[0]['model_name']}")
    print(f"Best validation loss: {best_configs.iloc[0]['best_val_loss']:.6f}")
    print(f"Mean validation loss: {df['best_val_loss'].mean():.6f}")
    print(f"Std validation loss: {df['best_val_loss'].std():.6f}")
    print("="*80 + "\n")

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
    
    if df.empty:
        print(f"No results found in {MODEL_FOLDER}. Make sure models have been trained and saved.")
        return
    
    print(f"Found {len(df)} trained models.")
    
    # Find best configurations
    print("\nFinding best configuration for each model type...")
    best_configs = find_best_configs(df)
    
    # Create diagnostics folder if it doesn't exist
    os.makedirs(DIAGNOSTICS_FOLDER, exist_ok=True)
    
    # Save best configurations
    best_configs.to_csv(OUTPUT_FILE, index=False)
    print(f"\nBest configurations saved to: {OUTPUT_FILE}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(df, best_configs)
    
    # Print summary
    print_summary(df, best_configs)
    
    # Save full results too
    full_results_file = Path(DIAGNOSTICS_FOLDER) / "all_results.csv"
    df.to_csv(full_results_file, index=False)
    print(f"Full results saved to: {full_results_file}")

if __name__ == "__main__":
    main()
