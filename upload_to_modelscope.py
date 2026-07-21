# upload_to_modelscope.py — 上传数据集和模型到 ModelScope
import os, sys
from modelscope.hub.api import HubApi

api = HubApi()
REPO = 'karos1214/compose-gnn'

def upload(local, remote, desc):
    if not os.path.exists(local):
        print(f'[SKIP] {desc}: not found')
        return
    size_mb = os.path.getsize(local) / (1024*1024)
    print(f'[{desc}] {size_mb:.0f}MB → {remote}')
    try:
        api.upload_file(repo_id=REPO, path_or_fileobj=local,
                        path_in_repo=remote, repo_type='model')
        print(f'[OK] {desc}')
    except Exception as e:
        print(f'[FAIL] {desc}: {e}')

BASE = os.path.dirname(os.path.abspath(__file__))

# ====== 数据集 ======
DS = 'datasets/03_graph'
upload(os.path.join(BASE, DS, 'merged_25cases_continuous_augmented_x7.pt'),
       'datasets/merged_25cases_continuous_augmented_x7.pt',
       'Main dataset (continuous, augmented x7)')

upload(os.path.join(BASE, DS, 'merged_25cases_continuous.pt'),
       'datasets/merged_25cases_continuous.pt',
       'Continuous dataset (no augmentation)')

# ====== 最佳模型 (v3 Hinge — 当前最优) ======
M = 'trained_model/2026_07_17_2013/GANv3_gat_hinge_bs2_Gh128_Dh64'
upload(os.path.join(BASE, M, 'best_model_sing.pth'),  'models/v3_hinge/best_sing.pth',  'v3 Hinge best sing')
upload(os.path.join(BASE, M, 'best_model_edge.pth'),  'models/v3_hinge/best_edge.pth',  'v3 Hinge best edge')
upload(os.path.join(BASE, M, 'final_model.pth'),      'models/v3_hinge/final.pth',       'v3 Hinge final')

# ====== v4 最佳模型 ======
M4 = 'trained_model/2026_07_18_0111/GANv4_gat_hinge_FocalDice_EMA_DE0.1_wup50'
upload(os.path.join(BASE, M4, 'best_model_sing_ema.pth'), 'models/v4_hinge/best_sing_ema.pth', 'v4 Hinge best sing EMA')
upload(os.path.join(BASE, M4, 'best_model_edge_ema.pth'), 'models/v4_hinge/best_edge_ema.pth', 'v4 Hinge best edge EMA')
upload(os.path.join(BASE, M4, 'final_model.pth'),         'models/v4_hinge/final.pth',          'v4 Hinge final')

# ====== v5 GNN 对比最佳模型 ======
M5_COMPARE = 'trained_model/v5_compare/2026_07_20_2202/v5_gat_hinge'
upload(os.path.join(BASE, M5_COMPARE, 'best_sing.pth'), 'models/v5_compare/best_sing.pth', 'v5 GAT Compare best sing (0.4085)')
upload(os.path.join(BASE, M5_COMPARE, 'best_edge.pth'), 'models/v5_compare/best_edge.pth', 'v5 GAT Compare best edge (0.3158)')

# ====== v5 物理约束最佳模型 ======
M5_PHYS = 'trained_model/v5_physics'
for ts_dir in sorted(os.listdir(os.path.join(BASE, M5_PHYS)), reverse=True):
    ts_path = os.path.join(BASE, M5_PHYS, ts_dir)
    if not os.path.isdir(ts_path):
        continue
    for exp_dir in sorted(os.listdir(ts_path)):
        exp_path = os.path.join(ts_path, exp_dir)
        if not os.path.isdir(exp_path):
            continue
        # Extract config name
        cfg = exp_dir.replace('v5_gat_physics_', '')
        for model_file in ['best_sing.pth', 'best_edge.pth']:
            local_path = os.path.join(exp_path, model_file)
            remote_path = f'models/v5_physics/{cfg}/{model_file}'
            if os.path.exists(local_path):
                upload(local_path, remote_path, f'v5 Physics {cfg} {model_file}')
    break  # Only upload latest timestamp

print('\n=== Upload complete! ===')
print(f'View at: https://modelscope.cn/models/{REPO}')
