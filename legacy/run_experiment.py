# run_experiment.py
# 使用本地数据集路径启动 4 个模型的完整训练
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from train_new import train_stress_flow
from models.test_models import PureSAGEModel, PureDynamicGATModel, PureGCNModel, PureClassicGATModel

# 使用本地数据集
DATA_PATH = "datasets/03_graph/merged_25cases_augmented_x7.pt"

all_models = [
    PureSAGEModel,
    PureDynamicGATModel,
    PureGCNModel,
    PureClassicGATModel,
]

common_params = {
    'epochs': 200,
    'lr': 1e-3,
    'batch_size': 2,
    'hidden_dim1': 128,
    'hidden_dim2': 64,
    'data_path': DATA_PATH,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_root': './trained_model',
    'seed': 42,
    'use_augmentation': False,  # augmented_x7 已是静态增强数据集
}

for model_cls in all_models:
    print("\n" + "="*60)
    print(f"Start training: {model_cls.__name__}")
    print("="*60)
    train_stress_flow(model_cls, **common_params)
