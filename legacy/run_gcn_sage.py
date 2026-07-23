#!/usr/bin/env python
"""Run GCN and SAGE comparison experiments (GAT already completed)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_gan_v5_compare import train_compare
import torch

DATA = 'datasets/03_graph/merged_25cases_continuous_augmented_x7.pt'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

results = {}

# GAT completed from earlier run: Best Sing Dice=0.4085, Edge Dice=0.3158
results['gat'] = {'sing_dice': 0.4085, 'edge_dice': 0.3158}

for gnn in ['gcn', 'sage']:
    print(f"\n{'='*60}")
    print(f"GNN: {gnn.upper()}")
    print(f"{'='*60}")
    _, _, _, best_s, best_e = train_compare(DATA, gnn_type=gnn, epochs=300, device=DEV)
    results[gnn] = {'sing_dice': best_s, 'edge_dice': best_e}

print(f"\n{'='*60}")
print("FINAL COMPARISON RESULTS")
print(f"{'='*60}")
for gnn, r in sorted(results.items(), key=lambda x: x[1]['sing_dice'], reverse=True):
    print(f"  {gnn.upper():>6s}: Sing Dice={r['sing_dice']:.4f}  Edge Dice={r['edge_dice']:.4f}")
