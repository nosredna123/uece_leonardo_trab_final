#!/usr/bin/env python3
"""
Generate feature importance visualization with population feature highlighted.
This creates a bar chart showing Random Forest and Gradient Boosting feature importances.
"""

import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# Load models
print("Loading trained models...")
rf_model = pickle.load(open('models/transit_coverage/random_forest.pkl', 'rb'))
gb_model = pickle.load(open('models/transit_coverage/gradient_boosting.pkl', 'rb'))

# Load feature names from metadata
import json
with open('models/transit_coverage/model_metadata.json', 'r') as f:
    metadata = json.load(f)
    feature_names = metadata['feature_names']

# Extract feature importances
rf_importance = rf_model.feature_importances_
gb_importance = gb_model.feature_importances_

# Create DataFrame
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Random Forest': rf_importance,
    'Gradient Boosting': gb_importance
})

# Sort by Random Forest importance
importance_df = importance_df.sort_values('Random Forest', ascending=False)

print("\nFeature Importances:")
print(importance_df.to_string(index=False))

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Define colors - highlight population in orange, others in blue
colors_rf = ['#FF8C00' if feat == 'population' else '#4A90E2' 
             for feat in importance_df['Feature']]
colors_gb = ['#FF8C00' if feat == 'population' else '#4A90E2' 
             for feat in importance_df['Feature']]

# Plot 1: Random Forest
y_pos = np.arange(len(importance_df))
ax1.barh(y_pos, importance_df['Random Forest'] * 100, color=colors_rf, alpha=0.8)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(importance_df['Feature'], fontsize=11)
ax1.set_xlabel('Importância (%)', fontsize=12, fontweight='bold')
ax1.set_title('Random Forest - Importância de Features', fontsize=14, fontweight='bold', pad=20)
ax1.invert_yaxis()
ax1.grid(axis='x', alpha=0.3)

# Add percentage labels
for i, (idx, row) in enumerate(importance_df.iterrows()):
    value = row['Random Forest'] * 100
    ax1.text(value + 0.5, i, f'{value:.2f}%', va='center', fontsize=10)

# Plot 2: Gradient Boosting
ax2.barh(y_pos, importance_df['Gradient Boosting'] * 100, color=colors_gb, alpha=0.8)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(importance_df['Feature'], fontsize=11)
ax2.set_xlabel('Importância (%)', fontsize=12, fontweight='bold')
ax2.set_title('Gradient Boosting - Importância de Features', fontsize=14, fontweight='bold', pad=20)
ax2.invert_yaxis()
ax2.grid(axis='x', alpha=0.3)

# Add percentage labels
for i, (idx, row) in enumerate(importance_df.iterrows()):
    value = row['Gradient Boosting'] * 100
    ax2.text(value + 0.5, i, f'{value:.2f}%', va='center', fontsize=10)

# Add legend for color coding
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#FF8C00', label='population (Feature Populacional)', alpha=0.8),
    Patch(facecolor='#4A90E2', label='Outras Features', alpha=0.8)
]
fig.legend(handles=legend_elements, loc='lower center', ncol=2, 
           fontsize=12, frameon=True, bbox_to_anchor=(0.5, -0.05))

plt.tight_layout(rect=[0, 0.03, 1, 1])

# Save figure
output_path = Path('reports/figures/population_feature_importance.png')
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✅ Visualization saved to: {output_path}")

# Also create a comparison bar chart
fig2, ax = plt.subplots(figsize=(14, 8))

# Prepare data
x = np.arange(len(importance_df))
width = 0.35

# Create bars
bars1 = ax.bar(x - width/2, importance_df['Random Forest'] * 100, width, 
               label='Random Forest', alpha=0.8, color='#4A90E2')
bars2 = ax.bar(x + width/2, importance_df['Gradient Boosting'] * 100, width,
               label='Gradient Boosting', alpha=0.8, color='#34C759')

# Highlight population bars in orange
for i, feat in enumerate(importance_df['Feature']):
    if feat == 'population':
        bars1[i].set_color('#FF8C00')
        bars2[i].set_color('#FF6B35')

# Labels and title
ax.set_xlabel('Features', fontsize=12, fontweight='bold')
ax.set_ylabel('Importância (%)', fontsize=12, fontweight='bold')
ax.set_title('Comparação de Importância de Features: Random Forest vs Gradient Boosting\n(Feature Populacional destacada em laranja)', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(importance_df['Feature'], rotation=45, ha='right', fontsize=11)
ax.legend(fontsize=12, loc='upper right')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()

# Save comparison figure
output_path2 = Path('reports/figures/population_feature_importance_comparison.png')
plt.savefig(output_path2, dpi=300, bbox_inches='tight')
print(f"✅ Comparison visualization saved to: {output_path2}")

plt.show()

print("\n📊 Feature importance analysis complete!")
print(f"\n🎯 Population feature importance:")
print(f"   - Random Forest: {importance_df[importance_df['Feature']=='population']['Random Forest'].values[0]*100:.2f}%")
print(f"   - Gradient Boosting: {importance_df[importance_df['Feature']=='population']['Gradient Boosting'].values[0]*100:.2f}%")
