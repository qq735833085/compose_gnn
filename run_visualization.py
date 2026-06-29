# run_visualization.py — 连续旋转模式可视化生成脚本
import sys, os, torch, math, random
import numpy as np

sys.path.insert(0, 'd:/composite_0602')
os.chdir('d:/composite_0602')

from utils.augment import StressFieldAugmentor, AUGMENT_CONFIG
from utils.visualization import (
    visualize_dashboard, visualize_augmentation_comparison,
    plot_stress_direction_field, plot_heatmap, plot_mesh_topology,
    plot_edge_heatmap, plot_label_distributions, plot_feature_histograms,
    plot_magnitude_comparison
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load real data
dataset = torch.load('graph_dataset/20260602_104_pro.pt', weights_only=False)
orig = dataset[0]
print(f'Sample: {orig.x.shape[0]} nodes, {orig.edge_index.shape[1]//2} edges')

output_dir = './visualization_output'
os.makedirs(output_dir, exist_ok=True)

# Verify continuous mode
aug = StressFieldAugmentor(AUGMENT_CONFIG)
print(f'Rotation mode: {aug.config["rotation"]["angle_mode"]}')

# Generate 4 augmented variants
random.seed(42)
np.random.seed(42)
variants = [aug(orig.clone()) for _ in range(4)]
labels = [f'Variant #{i+1}' for i in range(4)]

# ============================================================
# 1. Augmentation comparison (5 rows x 4 cols)
# ============================================================
print('1/7 Generating augmentation comparison...')
visualize_augmentation_comparison(
    orig, variants, labels=labels,
    save_path=os.path.join(output_dir, 'continuous_rotation_comparison.png'),
    dpi=150
)

# ============================================================
# 2. Direction field comparison
# ============================================================
print('2/7 Generating direction field comparison...')
fig, axes = plt.subplots(1, 3, figsize=(28, 9))
plot_stress_direction_field(orig, ax=axes[0], title='Original Stress Directions', arrow_spacing=600)
plot_stress_direction_field(variants[0], ax=axes[1], title='Variant #1 (continuous rot)', arrow_spacing=600)
plot_stress_direction_field(variants[1], ax=axes[2], title='Variant #2 (continuous rot)', arrow_spacing=600)
fig.suptitle('Stress Direction Field: Original vs Continuous Rotation Variants', fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(output_dir, 'continuous_rotation_direction_fields.png'), dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)

# ============================================================
# 3. Singularity heatmap comparison
# ============================================================
print('3/7 Generating singularity comparison...')
fig, axes = plt.subplots(1, 3, figsize=(28, 9))
for ax, data, t in zip(axes, [orig, variants[0], variants[2]], ['Original', 'Variant #1', 'Variant #2']):
    plot_heatmap(data, 'y_node', ax=ax, title=f'{t}: Singularity', cmap='hot', vmin=0, vmax=1, point_size=1.5)
fig.suptitle('Singularity: Original vs Continuous Rotation', fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(output_dir, 'continuous_rotation_singularity.png'), dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)

# ============================================================
# 4. m1_abs heatmap comparison
# ============================================================
print('4/7 Generating m1_abs comparison...')
fig, axes = plt.subplots(1, 3, figsize=(28, 9))
for ax, data, t in zip(axes, [orig, variants[0], variants[2]], ['Original', 'Variant #1', 'Variant #2']):
    plot_heatmap(data, 4, ax=ax, title=f'{t}: m1_abs', cmap='viridis', point_size=1.5)
fig.suptitle('m1_abs Magnitude: Original vs Continuous Rotation', fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(output_dir, 'continuous_rotation_m1abs.png'), dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)

# ============================================================
# 5. Dashboard for augmented variant
# ============================================================
print('5/7 Generating augmented dashboard...')
visualize_dashboard(variants[0], save_path=os.path.join(output_dir, 'continuous_rotation_dashboard.png'),
                     suptitle='Slab Stress Field Dashboard (Continuous Rotation Augmented)')

# ============================================================
# 6. Feature histogram comparison
# ============================================================
print('6/7 Generating feature histogram comparison...')
fig, axes = plt.subplots(2, 6, figsize=(20, 7))
x_orig = orig.x.cpu().numpy()
x_aug = variants[0].x.cpu().numpy()
fnames = ['x','y','m1_vx','m1_vy','m1_abs','m1_+t','m1_-c','m2_vx','m2_vy','m2_abs','m2_+t','m2_-c']
for i in range(12):
    r, c = divmod(i, 6)
    ax = axes[r, c]
    ax.hist(x_orig[:, i], bins=40, alpha=0.5, color='steelblue', density=True, label='Original', edgecolor='white', linewidth=0.2)
    ax.hist(x_aug[:, i], bins=40, alpha=0.5, color='coral', density=True, label='Augmented', edgecolor='white', linewidth=0.2)
    ax.set_title(fnames[i], fontsize=8); ax.tick_params(labelsize=6)
    if i == 0: ax.legend(fontsize=7)
fig.suptitle('Feature Distribution: Original vs Continuous Rotation', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(output_dir, 'continuous_rotation_feature_hists_compare.png'), dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)

# ============================================================
# 7. Coordinate distribution check (verify no clamping artifacts)
# ============================================================
print('7/7 Generating coordinate distribution check...')
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for idx, label in enumerate(['Original', 'Variant #1', 'Variant #2']):
    data = [orig, variants[0], variants[2]][idx]
    xn = data.x[:, 0].cpu().numpy()
    yn = data.x[:, 1].cpu().numpy()

    # Top row: x-y scatter
    ax = axes[0, idx]
    ax.scatter(xn, yn, s=0.3, alpha=0.5, c='steelblue', marker='.')
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_aspect('equal')
    ax.set_title(f'{label}: Node Positions')
    ax.set_xlabel('x'); ax.set_ylabel('y')

    # Bottom row: x histogram
    ax = axes[1, idx]
    ax.hist(xn, bins=80, color='steelblue', alpha=0.7, edgecolor='white', linewidth=0.2)
    ax.axvline(0, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axvline(1, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_title(f'{label}: x-coordinate Distribution')
    ax.set_xlabel('x')
    # Count boundary nodes (within 0.005 of edges)
    at_edge = ((xn <= 0.005) | (xn >= 0.995) | (yn <= 0.005) | (yn >= 0.995))
    pct = at_edge.sum() / len(xn) * 100
    ax.text(0.95, 0.95, f'boundary: {pct:.1f}%', transform=ax.transAxes, ha='right', va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

fig.suptitle('Coordinate Distribution: Continuous Rotation (Rescale, No Clamping)', fontsize=13, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(output_dir, 'continuous_rotation_coord_check.png'), dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)

# Report
print()
print('=== All 7 visualizations generated ===')
print(f'Output: {os.path.abspath(output_dir)}/')
for f in sorted(os.listdir(output_dir)):
    size_kb = os.path.getsize(os.path.join(output_dir, f)) / 1024
    print(f'  {f:50s} {size_kb:8.1f} KB')
