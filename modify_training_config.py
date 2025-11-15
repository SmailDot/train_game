"""
自動修改 utils/training_config.py 應用參數改進
"""

import os
import shutil
from datetime import datetime

print("=" * 80)
print("🔧 修改訓練配置文件")
print("=" * 80)

config_file = "utils/training_config.py"
backup_file = (
    f"utils/training_config.py.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)

# 備份原始文件
print(f"\n1. 備份原始配置...")
shutil.copy2(config_file, backup_file)
print(f"   ✅ 備份到: {backup_file}")

# 讀取原始文件
print(f"\n2. 讀取原始配置...")
with open(config_file, "r", encoding="utf-8") as f:
    content = f.read()

# 應用改進
print(f"\n3. 應用參數改進...")

improvements = [
    # GPU 配置改進
    ('    "lr": 2.5e-4,', '    "lr": 1e-4,  # 降低 from 2.5e-4 以減少震盪'),
    ('    "clip_eps": 0.2,', '    "clip_eps": 0.1,  # 降低 from 0.2 以限制更新幅度'),
    ('    "vf_coef": 0.5,', '    "vf_coef": 1.0,  # 增加 from 0.5 以加強 critic 訓練'),
    ('    "ent_coef": 0.01,', '    "ent_coef": 0.02,  # 增加 from 0.01 以增加探索'),
    (
        '    "max_grad_norm": 0.5,',
        '    "max_grad_norm": 0.3,  # 降低 from 0.5 以更強梯度裁剪',
    ),
    # CPU 配置改進
    ('    "lr": 3e-4,', '    "lr": 1e-4,  # 降低 from 3e-4 以減少震盪'),
    ('    "ent_coef": 0.05,', '    "ent_coef": 0.02,  # 調整 from 0.05 以平衡探索'),
]

changes_applied = 0
for old_str, new_str in improvements:
    if old_str in content:
        content = content.replace(old_str, new_str, 1)
        changes_applied += 1
        print(f"   ✅ {old_str.strip()}")
        print(f"      → {new_str.strip()}")

# 添加 weight_decay 說明（在文件頂部添加註釋）
header_addition = '''"""
參數改進歷史:
- 2025-11-15: 根據檢查點參數分析（#5940-#14460）應用改進
  - 主要問題: Critic bias 不穩定（CV 41.5%）
  - 改進: 降低學習率、增強 critic 訓練、更強梯度裁剪
  - 注意: weight_decay 需要在 pytorch_trainer.py 中的 optimizer 初始化時添加
"""

'''

# 在第一個註釋後插入
import_pos = content.find('"""') + 3
if import_pos > 3:
    import_pos = content.find('"""', import_pos) + 3
    content = content[:import_pos] + "\n" + header_addition + content[import_pos:]
    changes_applied += 1

print(f"\n   共應用 {changes_applied} 個改進")

# 保存修改後的文件
print(f"\n4. 保存修改後的配置...")
with open(config_file, "w", encoding="utf-8") as f:
    f.write(content)

print(f"   ✅ 已保存到: {config_file}")

# 修改 pytorch_trainer.py 添加 weight_decay
print(f"\n5. 修改 pytorch_trainer.py 添加 weight_decay...")

trainer_file = "agents/pytorch_trainer.py"
trainer_backup = (
    f"agents/pytorch_trainer.py.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)

# 備份
shutil.copy2(trainer_file, trainer_backup)
print(f"   ✅ 備份到: {trainer_backup}")

# 讀取文件
with open(trainer_file, "r", encoding="utf-8") as f:
    trainer_content = f.read()

# 查找並替換 optimizer 初始化
old_optimizer_line = "self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)"
new_optimizer_line = (
    "self.opt = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=1e-4)"
)

if old_optimizer_line in trainer_content:
    trainer_content = trainer_content.replace(old_optimizer_line, new_optimizer_line)
    print(f"   ✅ 添加 weight_decay=1e-4 到 Adam optimizer")

    # 保存
    with open(trainer_file, "w", encoding="utf-8") as f:
        f.write(trainer_content)
    print(f"   ✅ 已保存修改")
else:
    print(f"   ⚠️  未找到標準的 optimizer 初始化，需要手動添加")

# === 生成對比報告 ===
print(f"\n" + "=" * 80)
print(f"📊 修改總結")
print(f"=" * 80)

print(
    f"""
已修改的配置參數:
1. learning_rate: 2.5e-4 → 1e-4 (GPU), 3e-4 → 1e-4 (CPU)
2. clip_eps: 0.2 → 0.1
3. vf_coef: 0.5 → 1.0
4. ent_coef: 0.01 → 0.02 (GPU), 0.05 → 0.02 (CPU)
5. max_grad_norm: 0.5 → 0.3
6. weight_decay: 0 → 1e-4 (在 optimizer 中)

原因:
- 參數分析顯示 critic.bias 變異係數達 41.5%
- 多個參數在訓練期間持續增長
- Actor bias 在崩潰點有 5% 跳變

預期效果:
✓ Critic 更穩定（CV < 20%）
✓ 參數變化更平穩
✓ 訓練更穩定，減少崩潰風險

備份文件:
- {backup_file}
- {trainer_backup}

下一步:
1. python execute_complete_fix.py  # 回檔到 checkpoint_5930.pt
2. python run_game.py              # 啟動訓練（自動使用新配置）
3. 監控 training_history.json 和分數變化
"""
)

print(f"=" * 80)
print(f"✅ 配置修改完成！")
print(f"=" * 80)
