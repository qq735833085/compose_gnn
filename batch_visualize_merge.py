# batch_visualize_merge.py — 批量生成所有案例的 merge_check 图
import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from visualize_merge_check import visualize_merge, parse_file_pairs, DATA_DIR

# 输出目录
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'merge_checks')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 找到所有案例
all_files = glob.glob(os.path.join(DATA_DIR, '*.csv'))
nodes_files = [os.path.basename(f) for f in all_files if 'nodes' in os.path.basename(f)]
pairs, errors = parse_file_pairs(nodes_files)

print(f'Found {len(pairs)} cases, generating merge_check for each...')
print(f'Output: {OUTPUT_DIR}/')
print()

success = 0
failed = []

for case_name in sorted(pairs.keys()):
    versions = pairs[case_name]
    save_path = os.path.join(OUTPUT_DIR, f'{case_name}_merge_check.png')

    # 跳过已生成的
    if os.path.exists(save_path):
        print(f'  [{case_name}] already exists, skipping')
        success += 1
        continue

    try:
        visualize_merge(case_name, versions, DATA_DIR,
                       os.path.join(os.path.dirname(os.path.abspath(__file__)), 'merged_data'),
                       save_path=save_path)
        success += 1
        print(f'  [{case_name}] OK ({success}/{len(pairs)})')
    except Exception as e:
        failed.append((case_name, str(e)))
        print(f'  [{case_name}] FAILED: {e}')

print()
print(f'{"="*60}')
print(f'Done: {success}/{len(pairs)} generated, {len(failed)} failed')
if failed:
    print('Failures:')
    for name, err in failed:
        print(f'  {name}: {err}')
print(f'Output: {os.path.abspath(OUTPUT_DIR)}/')
