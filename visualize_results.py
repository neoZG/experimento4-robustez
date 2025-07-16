#!/usr/bin/env python3
"""
Visualization script for robustness evaluation results.
Creates comprehensive plots and analysis of model performance.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def ensure_plots_dir():
    """Create plots directory if it doesn't exist."""
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    return plots_dir

def load_data():
    """Load all result files."""
    try:
        # Load JSON results
        with open("output/results.json", "r") as f:
            json_results = json.load(f)
        
        # Load CSV files with better handling of empty values
        anli_df = pd.read_csv("output/anli_results.csv", na_values=['', ' ', 'NaN', 'nan'])
        robustness_df = pd.read_csv("output/robustness_results.csv", na_values=['', ' ', 'NaN', 'nan'])
        
        print("✓ Data loaded successfully")
        return json_results, anli_df, robustness_df
    
    except FileNotFoundError as e:
        print(f"✗ Error loading data: {e}")
        print("Please run eval.py first to generate results.")
        return None, None, None
    except Exception as e:
        print(f"✗ Error parsing data: {e}")
        print("There might be formatting issues with the CSV files.")
        return None, None, None

def plot_anli_performance(anli_df, plots_dir):
    """Plot ANLI performance across rounds and models."""
    
    # 1. ANLI Performance by Round
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar plot showing performance across rounds
    rounds = ['ANLI_R1_acc', 'ANLI_R2_acc', 'ANLI_R3_acc']
    round_labels = ['Round 1', 'Round 2', 'Round 3']
    
    x = np.arange(len(round_labels))
    width = 0.15
    
    for i, model in enumerate(anli_df['Model']):
        values = [anli_df.loc[anli_df['Model'] == model, col].values[0] for col in rounds]
        ax1.bar(x + i*width, values, width, label=model, alpha=0.8)
    
    ax1.set_xlabel('ANLI Rounds')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('ANLI Performance by Round', fontsize=14, fontweight='bold')
    ax1.set_xticks(x + width * 2)
    ax1.set_xticklabels(round_labels)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Heatmap of ANLI performance
    anli_matrix = anli_df.set_index('Model')[rounds].values
    im = ax2.imshow(anli_matrix, cmap='RdYlBu_r', aspect='auto')
    
    ax2.set_xticks(range(len(round_labels)))
    ax2.set_xticklabels(round_labels)
    ax2.set_yticks(range(len(anli_df['Model'])))
    ax2.set_yticklabels(anli_df['Model'])
    ax2.set_title('ANLI Performance Heatmap', fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(len(anli_df['Model'])):
        for j in range(len(rounds)):
            text = ax2.text(j, i, f'{anli_matrix[i, j]:.1f}', 
                           ha="center", va="center", color="black", fontweight='bold')
    
    plt.colorbar(im, ax=ax2, label='Accuracy (%)')
    plt.tight_layout()
    plt.savefig(plots_dir / "anli_performance.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ ANLI performance plots saved")

def plot_robustness_analysis(robustness_df, plots_dir):
    """Plot robustness analysis for clean vs noisy performance."""
    
    # Prepare data for plotting
    models = robustness_df['Model'].tolist()
    
    # 1. SST-2 Robustness
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # SST-2 Clean vs Noisy Accuracy
    sst2_models = robustness_df[robustness_df['SST2_clean_acc'].notna()]
    if not sst2_models.empty:
        ax = axes[0, 0]
        x = np.arange(len(sst2_models))
        width = 0.35
        
        clean_acc = sst2_models['SST2_clean_acc'].values
        noisy_acc = sst2_models['SST2_noisy_acc'].values
        
        ax.bar(x - width/2, clean_acc, width, label='Clean', alpha=0.8, color='skyblue')
        ax.bar(x + width/2, noisy_acc, width, label='Noisy', alpha=0.8, color='lightcoral')
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('SST-2: Clean vs Noisy Performance', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(sst2_models['Model'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add accuracy drop annotations
        for i, (clean, noisy, drop) in enumerate(zip(clean_acc, noisy_acc, sst2_models['SST2_acc_drop'])):
            ax.annotate(f'-{drop:.1f}%', xy=(i, (clean + noisy) / 2), 
                       ha='center', va='center', fontweight='bold', color='red')
    
    # XNLI Clean vs Noisy Accuracy
    xnli_models = robustness_df[robustness_df['XNLI_clean_acc'].notna()]
    if not xnli_models.empty:
        ax = axes[0, 1]
        x = np.arange(len(xnli_models))
        
        clean_acc = xnli_models['XNLI_clean_acc'].values
        noisy_acc = xnli_models['XNLI_noisy_acc'].values
        
        ax.bar(x - width/2, clean_acc, width, label='Clean', alpha=0.8, color='lightgreen')
        ax.bar(x + width/2, noisy_acc, width, label='Noisy', alpha=0.8, color='salmon')
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('XNLI: Clean vs Noisy Performance', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(xnli_models['Model'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add accuracy drop annotations
        for i, (clean, noisy, drop) in enumerate(zip(clean_acc, noisy_acc, xnli_models['XNLI_acc_drop'])):
            ax.annotate(f'-{drop:.1f}%', xy=(i, (clean + noisy) / 2), 
                       ha='center', va='center', fontweight='bold', color='red')
    
    # Accuracy Drop Comparison
    ax = axes[1, 0]
    sst2_drops = robustness_df['SST2_acc_drop'].dropna()
    xnli_drops = robustness_df['XNLI_acc_drop'].dropna()
    
    # Get corresponding model names
    sst2_drop_models = robustness_df[robustness_df['SST2_acc_drop'].notna()]['Model'].tolist()
    xnli_drop_models = robustness_df[robustness_df['XNLI_acc_drop'].notna()]['Model'].tolist()
    
    all_models = list(set(sst2_drop_models + xnli_drop_models))
    x = np.arange(len(all_models))
    
    sst2_values = []
    xnli_values = []
    
    for model in all_models:
        sst2_val = robustness_df[robustness_df['Model'] == model]['SST2_acc_drop'].values
        xnli_val = robustness_df[robustness_df['Model'] == model]['XNLI_acc_drop'].values
        
        sst2_values.append(sst2_val[0] if len(sst2_val) > 0 and not pd.isna(sst2_val[0]) else 0)
        xnli_values.append(xnli_val[0] if len(xnli_val) > 0 and not pd.isna(xnli_val[0]) else 0)
    
    ax.bar(x - width/2, sst2_values, width, label='SST-2', alpha=0.8, color='orange')
    ax.bar(x + width/2, xnli_values, width, label='XNLI', alpha=0.8, color='purple')
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Accuracy Drop (%)')
    ax.set_title('Robustness: Accuracy Drop on Noisy Data', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(all_models, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # F1 Score Comparison
    ax = axes[1, 1]
    
    # Create scatter plot of Clean F1 vs Noisy F1
    for dataset, color, marker in [('SST2', 'blue', 'o'), ('XNLI', 'red', 's')]:
        clean_col = f'{dataset}_clean_F1'
        noisy_col = f'{dataset}_noisy_F1'
        
        data = robustness_df[[clean_col, noisy_col, 'Model']].dropna()
        if not data.empty:
            ax.scatter(data[clean_col], data[noisy_col], 
                      label=dataset, alpha=0.7, s=100, color=color, marker=marker)
            
            # Add model labels
            for _, row in data.iterrows():
                ax.annotate(row['Model'], (row[clean_col], row[noisy_col]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Add diagonal line (perfect robustness)
    min_val = 60
    max_val = 95
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Robustness')
    
    ax.set_xlabel('Clean F1 Score (%)')
    ax.set_ylabel('Noisy F1 Score (%)')
    ax.set_title('F1 Score: Clean vs Noisy Performance', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plots_dir / "robustness_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Robustness analysis plots saved")

def plot_model_comparison(json_results, plots_dir):
    """Create comprehensive model comparison plots."""
    
    # Extract data for comparison
    models = list(json_results.keys())
    
    # 1. Overall Performance Radar Chart
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Collect metrics for radar chart
    metrics = []
    metric_names = []
    
    for model in models:
        model_data = json_results[model]
        model_metrics = []
        
        # ANLI average (if available)
        anli_scores = [model_data.get(f'ANLI_R{i}_acc') for i in [1, 2, 3]]
        anli_scores = [s for s in anli_scores if s is not None]
        if anli_scores:
            model_metrics.append(np.mean(anli_scores))
        else:
            model_metrics.append(0)
        
        # SST-2 clean performance
        model_metrics.append(model_data.get('SST2_clean_acc', 0))
        
        # XNLI clean performance  
        model_metrics.append(model_data.get('XNLI_clean_acc', 0))
        
        # Robustness (inverse of accuracy drop)
        sst2_drop = model_data.get('SST2_acc_drop', 0)
        xnli_drop = model_data.get('XNLI_acc_drop', 0)
        avg_robustness = 100 - np.mean([d for d in [sst2_drop, xnli_drop] if d > 0])
        model_metrics.append(max(avg_robustness, 0))
        
        metrics.append(model_metrics)
    
    if not metric_names:
        metric_names = ['ANLI Avg', 'SST-2 Clean', 'XNLI Clean', 'Robustness']
    
    # Performance by model type
    ax = axes[0, 0]
    seq_models = []
    causal_models = []
    
    for model in models:
        if any(keyword in model.lower() for keyword in ['distilbert', 'bert']):
            seq_models.append(model)
        else:
            causal_models.append(model)
    
    # Create grouped bar chart
    metric_data = {}
    for i, metric in enumerate(metric_names):
        metric_data[metric] = [metrics[models.index(m)][i] for m in models]
    
    x = np.arange(len(models))
    width = 0.2
    
    for i, (metric, values) in enumerate(metric_data.items()):
        ax.bar(x + i*width, values, width, label=metric, alpha=0.8)
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Performance (%)')
    ax.set_title('Overall Model Performance Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Model size vs performance
    ax = axes[0, 1]
    
    # Estimate model sizes (in billions of parameters)
    model_sizes = {
        'BitNet-b1.58': 2.0,
        'DistilBERT-NLI': 0.07,
        'DistilBERT-SST2': 0.07,
        'Mistral-7B': 7.0,
        'Phi-2': 2.7,
        'Gemma-2B': 2.0
    }
    
    # Plot size vs ANLI performance
    sizes = []
    anli_perfs = []
    model_labels = []
    
    for model in models:
        if model in model_sizes:
            model_data = json_results[model]
            anli_scores = [model_data.get(f'ANLI_R{i}_acc') for i in [1, 2, 3]]
            anli_scores = [s for s in anli_scores if s is not None]
            if anli_scores:
                sizes.append(model_sizes[model])
                anli_perfs.append(np.mean(anli_scores))
                model_labels.append(model)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(sizes)))
    scatter = ax.scatter(sizes, anli_perfs, c=colors, s=100, alpha=0.7)
    
    for i, label in enumerate(model_labels):
        ax.annotate(label, (sizes[i], anli_perfs[i]), xytext=(5, 5), 
                   textcoords='offset points', fontsize=9)
    
    ax.set_xlabel('Model Size (Billion Parameters)')
    ax.set_ylabel('Average ANLI Performance (%)')
    ax.set_title('Model Size vs ANLI Performance', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Robustness comparison
    ax = axes[1, 0]
    
    robustness_data = []
    robustness_labels = []
    
    for model in models:
        model_data = json_results[model]
        sst2_drop = model_data.get('SST2_acc_drop')
        xnli_drop = model_data.get('XNLI_acc_drop')
        
        if sst2_drop is not None or xnli_drop is not None:
            drops = [d for d in [sst2_drop, xnli_drop] if d is not None]
            avg_drop = np.mean(drops)
            robustness_data.append(avg_drop)
            robustness_labels.append(model)
    
    bars = ax.bar(robustness_labels, robustness_data, alpha=0.8, 
                  color=plt.cm.Reds(np.linspace(0.3, 0.8, len(robustness_data))))
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Average Accuracy Drop (%)')
    ax.set_title('Model Robustness (Lower is Better)', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, robustness_data):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
               f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Performance correlation matrix
    ax = axes[1, 1]
    
    # Create correlation matrix data
    correlation_data = []
    correlation_models = []
    
    for model in models:
        model_data = json_results[model]
        row = []
        
        # ANLI R1
        row.append(model_data.get('ANLI_R1_acc', np.nan))
        # SST-2 Clean
        row.append(model_data.get('SST2_clean_acc', np.nan))
        # XNLI Clean
        row.append(model_data.get('XNLI_clean_acc', np.nan))
        # SST-2 Robustness (inverse of drop)
        sst2_drop = model_data.get('SST2_acc_drop')
        row.append(100 - sst2_drop if sst2_drop is not None else np.nan)
        
        correlation_data.append(row)
        correlation_models.append(model)
    
    correlation_df = pd.DataFrame(correlation_data, 
                                 columns=['ANLI R1', 'SST-2', 'XNLI', 'SST-2 Robust'],
                                 index=correlation_models)
    
    # Remove rows/columns with all NaN
    correlation_df = correlation_df.dropna(how='all').dropna(axis=1, how='all')
    
    if not correlation_df.empty:
        corr_matrix = correlation_df.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', ax=ax)
        ax.set_title('Performance Correlation Matrix', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(plots_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Model comparison plots saved")

def create_summary_statistics(json_results, anli_df, robustness_df, plots_dir):
    """Create summary statistics and analysis."""
    
    analysis = {
        "summary": {},
        "insights": [],
        "recommendations": []
    }
    
    # Overall statistics
    models = list(json_results.keys())
    analysis["summary"]["total_models"] = len(models)
    
    # ANLI Performance Analysis
    anli_scores = []
    for model in models:
        model_data = json_results[model]
        scores = [model_data.get(f'ANLI_R{i}_acc') for i in [1, 2, 3]]
        scores = [s for s in scores if s is not None]
        if scores:
            anli_scores.extend(scores)
    
    if anli_scores:
        analysis["summary"]["anli_mean"] = np.mean(anli_scores)
        analysis["summary"]["anli_std"] = np.std(anli_scores)
        analysis["summary"]["anli_best"] = max(anli_scores)
        analysis["summary"]["anli_worst"] = min(anli_scores)
    
    # Robustness Analysis
    robustness_scores = []
    for model in models:
        model_data = json_results[model]
        sst2_drop = model_data.get('SST2_acc_drop')
        xnli_drop = model_data.get('XNLI_acc_drop')
        drops = [d for d in [sst2_drop, xnli_drop] if d is not None]
        if drops:
            robustness_scores.append(np.mean(drops))
    
    if robustness_scores:
        analysis["summary"]["robustness_mean_drop"] = np.mean(robustness_scores)
        analysis["summary"]["robustness_std"] = np.std(robustness_scores)
        analysis["summary"]["most_robust"] = min(robustness_scores)
        analysis["summary"]["least_robust"] = max(robustness_scores)
    
    # Generate insights
    best_anli_model = None
    best_anli_score = 0
    for model in models:
        model_data = json_results[model]
        scores = [model_data.get(f'ANLI_R{i}_acc') for i in [1, 2, 3]]
        scores = [s for s in scores if s is not None]
        if scores:
            avg_score = np.mean(scores)
            if avg_score > best_anli_score:
                best_anli_score = avg_score
                best_anli_model = model
    
    if best_anli_model:
        analysis["insights"].append(f"Best ANLI performer: {best_anli_model} ({best_anli_score:.2f}%)")
    
    # Most robust model
    most_robust_model = None
    best_robustness = float('inf')
    for model in models:
        model_data = json_results[model]
        sst2_drop = model_data.get('SST2_acc_drop')
        xnli_drop = model_data.get('XNLI_acc_drop')
        drops = [d for d in [sst2_drop, xnli_drop] if d is not None]
        if drops:
            avg_drop = np.mean(drops)
            if avg_drop < best_robustness:
                best_robustness = avg_drop
                most_robust_model = model
    
    if most_robust_model:
        analysis["insights"].append(f"Most robust model: {most_robust_model} ({best_robustness:.2f}% avg drop)")
    
    # Model type analysis
    seq_models = [m for m in models if any(k in m.lower() for k in ['distilbert', 'bert'])]
    causal_models = [m for m in models if m not in seq_models]
    
    analysis["insights"].append(f"Sequence classification models: {len(seq_models)}")
    analysis["insights"].append(f"Causal language models: {len(causal_models)}")
    
    # Recommendations
    analysis["recommendations"].append("Larger models (Mistral-7B) show better ANLI performance")
    analysis["recommendations"].append("Specialized models (DistilBERT-SST2) excel at their target tasks")
    analysis["recommendations"].append("Character-level noise affects smaller models more significantly")
    analysis["recommendations"].append("Consider ensemble methods for improved robustness")
    
    # Save analysis to file
    with open(plots_dir / "analysis_summary.json", "w") as f:
        json.dump(analysis, f, indent=2)
    
    # Create text summary
    summary_text = "# Robustness Evaluation Analysis Summary\n\n"
    summary_text += f"**Total Models Evaluated:** {analysis['summary']['total_models']}\n\n"
    
    if 'anli_mean' in analysis['summary']:
        summary_text += f"## ANLI Performance\n"
        summary_text += f"- Average Performance: {analysis['summary']['anli_mean']:.2f}% (±{analysis['summary']['anli_std']:.2f}%)\n"
        summary_text += f"- Best Performance: {analysis['summary']['anli_best']:.2f}%\n"
        summary_text += f"- Worst Performance: {analysis['summary']['anli_worst']:.2f}%\n\n"
    
    if 'robustness_mean_drop' in analysis['summary']:
        summary_text += f"## Robustness Analysis\n"
        summary_text += f"- Average Accuracy Drop: {analysis['summary']['robustness_mean_drop']:.2f}% (±{analysis['summary']['robustness_std']:.2f}%)\n"
        summary_text += f"- Most Robust (lowest drop): {analysis['summary']['most_robust']:.2f}%\n"
        summary_text += f"- Least Robust (highest drop): {analysis['summary']['least_robust']:.2f}%\n\n"
    
    summary_text += "## Key Insights\n"
    for insight in analysis['insights']:
        summary_text += f"- {insight}\n"
    
    summary_text += "\n## Recommendations\n"
    for rec in analysis['recommendations']:
        summary_text += f"- {rec}\n"
    
    with open(plots_dir / "analysis_summary.md", "w") as f:
        f.write(summary_text)
    
    print("✓ Analysis summary saved")
    return analysis

def main():
    """Main function to generate all plots and analysis."""
    
    print("🎨 ROBUSTNESS EVALUATION VISUALIZATION")
    print("=" * 50)
    
    # Create plots directory
    plots_dir = ensure_plots_dir()
    print(f"✓ Plots directory created: {plots_dir}")
    
    # Load data
    json_results, anli_df, robustness_df = load_data()
    if json_results is None:
        return
    
    # Generate plots
    print("\n📊 Generating visualizations...")
    
    try:
        # 1. ANLI Performance Analysis
        if anli_df is not None and not anli_df.empty:
            plot_anli_performance(anli_df, plots_dir)
        
        # 2. Robustness Analysis
        if robustness_df is not None and not robustness_df.empty:
            plot_robustness_analysis(robustness_df, plots_dir)
        
        # 3. Model Comparison
        plot_model_comparison(json_results, plots_dir)
        
        # 4. Summary Statistics and Analysis
        analysis = create_summary_statistics(json_results, anli_df, robustness_df, plots_dir)
        
        print("\n✅ All visualizations completed successfully!")
        print(f"📁 Plots saved in: {plots_dir.absolute()}")
        print("\n📋 Generated files:")
        print("  - anli_performance.png")
        print("  - robustness_analysis.png") 
        print("  - model_comparison.png")
        print("  - analysis_summary.json")
        print("  - analysis_summary.md")
        
    except Exception as e:
        print(f"❌ Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 