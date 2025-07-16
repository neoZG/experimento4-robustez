#!/usr/bin/env python3
"""
Simple script to display visualization results and analysis summary.
"""

import json
import os
from pathlib import Path

def display_results():
    """Display key results from the robustness evaluation."""
    
    print("📊 ROBUSTNESS EVALUATION RESULTS SUMMARY")
    print("=" * 60)
    
    # Check if plots directory exists
    plots_dir = Path("plots")
    if not plots_dir.exists():
        print("❌ No plots directory found. Run visualize_results.py first.")
        return
    
    # Display generated files
    print("\n📁 Generated Visualization Files:")
    for file in plots_dir.iterdir():
        if file.is_file():
            size_mb = file.stat().st_size / (1024 * 1024)
            if file.suffix == '.png':
                print(f"  🖼️  {file.name} ({size_mb:.1f} MB)")
            elif file.suffix == '.json':
                print(f"  📄 {file.name}")
            elif file.suffix == '.md':
                print(f"  📝 {file.name}")
    
    # Load and display analysis summary
    summary_file = plots_dir / "analysis_summary.json"
    if summary_file.exists():
        print("\n🔍 KEY FINDINGS:")
        with open(summary_file, 'r') as f:
            analysis = json.load(f)
        
        summary = analysis.get('summary', {})
        insights = analysis.get('insights', [])
        recommendations = analysis.get('recommendations', [])
        
        if summary:
            print(f"  • Total Models Evaluated: {summary.get('total_models', 'N/A')}")
            if 'anli_mean' in summary:
                print(f"  • Average ANLI Performance: {summary['anli_mean']:.2f}% (±{summary['anli_std']:.2f}%)")
                print(f"  • Best ANLI Performance: {summary['anli_best']:.2f}%")
            if 'robustness_mean_drop' in summary:
                print(f"  • Average Robustness Drop: {summary['robustness_mean_drop']:.2f}% (±{summary['robustness_std']:.2f}%)")
        
        if insights:
            print("\n💡 TOP INSIGHTS:")
            for insight in insights[:3]:  # Show top 3 insights
                print(f"  • {insight}")
        
        if recommendations:
            print("\n🎯 RECOMMENDATIONS:")
            for rec in recommendations[:3]:  # Show top 3 recommendations
                print(f"  • {rec}")
    
    # Display plots description
    print("\n🎨 VISUALIZATION DESCRIPTIONS:")
    plot_descriptions = {
        "anli_performance.png": "ANLI performance across rounds and models with heatmap",
        "robustness_analysis.png": "Clean vs noisy performance comparison and F1 analysis",
        "model_comparison.png": "Comprehensive model comparison including size vs performance"
    }
    
    for plot_file, description in plot_descriptions.items():
        if (plots_dir / plot_file).exists():
            print(f"  📈 {plot_file}: {description}")
    
    print("\n" + "=" * 60)
    print("✅ Analysis complete! Check the plots/ directory for detailed visualizations.")

if __name__ == "__main__":
    display_results() 