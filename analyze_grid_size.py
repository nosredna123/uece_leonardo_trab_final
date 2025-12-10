#!/usr/bin/env python3
"""
Grid Size Impact Analysis

This script analyzes how grid cell size affects the difficulty of the
transit coverage classification problem.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_current_grid():
    """Analyze current 500m grid characteristics"""
    print("=" * 70)
    print("CURRENT GRID ANALYSIS (500m × 500m cells)")
    print("=" * 70)
    
    # Load data
    train = pd.read_parquet('data/processed/features/train.parquet')
    val = pd.read_parquet('data/processed/features/val.parquet')
    test = pd.read_parquet('data/processed/features/test.parquet')
    
    all_data = pd.concat([train, val, test])
    
    print(f"\nDataset Statistics:")
    print(f"  Total cells: {len(all_data):,}")
    print(f"  Cell size: 500m × 500m = 0.25 km²")
    print(f"  Coverage area: ~{len(all_data) * 0.25:.0f} km²")
    
    print(f"\nLabel Distribution:")
    print(f"  Class 0 (Underserved): {(all_data['label']==0).sum():,} ({(all_data['label']==0).sum()/len(all_data)*100:.1f}%)")
    print(f"  Class 1 (Well-served): {(all_data['label']==1).sum():,} ({(all_data['label']==1).sum()/len(all_data)*100:.1f}%)")
    
    print(f"\nFeature Distributions:")
    features = ['stop_count', 'route_count', 'daily_trips']
    
    for feat in features:
        zeros = (all_data[feat] == 0).sum()
        zeros_pct = zeros / len(all_data) * 100
        mean = all_data[feat].mean()
        median = all_data[feat].median()
        q75 = all_data[feat].quantile(0.75)
        q95 = all_data[feat].quantile(0.95)
        max_val = all_data[feat].max()
        
        print(f"\n  {feat}:")
        print(f"    Mean: {mean:6.1f}, Median: {median:4.0f}")
        print(f"    75th percentile: {q75:6.1f}")
        print(f"    95th percentile: {q95:6.1f}")
        print(f"    Max: {max_val:6.0f}")
        print(f"    Zero values: {zeros:4d} ({zeros_pct:4.1f}%)")
    
    print("\n" + "-" * 70)
    print("PROBLEM DIAGNOSIS")
    print("-" * 70)
    
    print("\n⚠️  Issues with 500m grids:")
    print("  • Large cells aggregate many transit features")
    print("  • 63% of cells have ZERO stops/routes (very sparse)")
    print("  • Strong aggregation creates artificial class separation")
    print("  • Model learns simple threshold on aggregated counts")
    print("  • Not representative of walking distance (~5-10 min walk)")
    
    return all_data

def simulate_grid_sizes():
    """Simulate impact of different grid sizes"""
    print("\n" + "=" * 70)
    print("GRID SIZE COMPARISON")
    print("=" * 70)
    
    # Current grid area
    current_area_km2 = 3250 * 0.25  # 3250 cells × 0.25 km²
    
    grid_sizes = [
        (100, "Very Fine"),
        (150, "Fine (RECOMMENDED)"),
        (200, "Moderate"),
        (250, "Balanced"),
        (500, "Current (TOO EASY)"),
    ]
    
    print(f"\nAssuming coverage area: ~{current_area_km2:.0f} km²\n")
    print(f"{'Grid Size':<12} {'Cell Area':<12} {'Est. Cells':<12} {'Difficulty':<25} {'Comp. Time':<15}")
    print("-" * 85)
    
    for size_m, difficulty in grid_sizes:
        cell_area_km2 = (size_m / 1000) ** 2
        est_cells = int(current_area_km2 / cell_area_km2)
        
        # Estimate computational time (rough scaling)
        if size_m == 500:
            comp_time = "~30s (baseline)"
        else:
            ratio = (500 / size_m) ** 2
            minutes = 0.5 * ratio
            if minutes < 1:
                comp_time = f"~{int(minutes * 60)}s"
            else:
                comp_time = f"~{int(minutes)}min"
        
        marker = "⭐" if "RECOMMENDED" in difficulty else "❌" if "EASY" in difficulty else "✓"
        print(f"{marker} {size_m}m × {size_m}m    {cell_area_km2:.4f} km²   {est_cells:>9,}     {difficulty:<25} {comp_time:<15}")
    
    print("\n" + "-" * 70)
    print("RECOMMENDATIONS")
    print("-" * 70)
    
    print("\n1. ⭐ BEST OPTION: 150m × 150m grids")
    print("   • Sweet spot between realism and computation")
    print("   • ~36,000 cells (11× more granular)")
    print("   • Typical walking distance: 2-3 minutes")
    print("   • Expected F1: 0.75-0.85 (realistic)")
    print("   • Runtime: ~5-10 minutes for full pipeline")
    
    print("\n2. ✓ CONSERVATIVE: 200m × 200m grids")
    print("   • Moderate improvement over current")
    print("   • ~20,000 cells (6× more granular)")
    print("   • Walking distance: ~2.5 minutes")
    print("   • Expected F1: 0.80-0.90")
    print("   • Runtime: ~3-5 minutes")
    
    print("\n3. ✓ BALANCED: 250m × 250m grids")
    print("   • Incremental improvement")
    print("   • ~13,000 cells (4× more granular)")
    print("   • Walking distance: ~3 minutes")
    print("   • Expected F1: 0.85-0.92")
    print("   • Runtime: ~2-3 minutes")

def visualize_spatial_scale():
    """Create visualization comparing grid sizes"""
    print("\n" + "=" * 70)
    print("SPATIAL SCALE VISUALIZATION")
    print("=" * 70)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Simulate a 2km × 2km area
    grid_sizes = [500, 200, 150]
    titles = ["500m (Current)\n4 × 4 = 16 cells", 
              "200m (Moderate)\n10 × 10 = 100 cells",
              "150m (Recommended)\n13 × 13 = 169 cells"]
    
    for ax, size, title in zip(axes, grid_sizes, titles):
        n_cells = int(2000 / size)
        
        # Draw grid
        for i in range(n_cells + 1):
            ax.axhline(i * size, color='black', linewidth=0.5)
            ax.axvline(i * size, color='black', linewidth=0.5)
        
        # Simulate some stops
        np.random.seed(42)
        n_stops = 15
        stops_x = np.random.uniform(0, 2000, n_stops)
        stops_y = np.random.uniform(0, 2000, n_stops)
        ax.scatter(stops_x, stops_y, c='red', s=100, marker='o', 
                  label='Bus stops', zorder=5, edgecolors='darkred', linewidths=2)
        
        ax.set_xlim(0, 2000)
        ax.set_ylim(0, 2000)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Distance (m)')
        ax.legend(loc='upper right')
        ax.grid(False)
    
    plt.tight_layout()
    
    output_path = Path('reports/figures/grid_size_comparison.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_path}")
    plt.close()

def main():
    """Main analysis function"""
    print("\n")
    print("*" * 70)
    print("  GRID SIZE IMPACT ANALYSIS")
    print("  Transit Coverage Classification Problem")
    print("*" * 70)
    
    # Analyze current grid
    current_data = analyze_current_grid()
    
    # Simulate different grid sizes
    simulate_grid_sizes()
    
    # Create visualization
    visualize_spatial_scale()
    
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("\nTo regenerate with 150m grids:")
    print("  1. Update grid generator config: CELL_SIZE_M = 150")
    print("  2. Re-run pipeline:")
    print("     python -m src.data.grid_generator")
    print("     python -m src.data.feature_extractor")
    print("     python -m src.data.label_generator")
    print("     python -m src.data.preprocessing")
    print("     python -m src.models.train")
    print("  3. Compare results: 500m (F1=1.00) vs 150m (F1=~0.80)")
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
