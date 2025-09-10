# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Subset

############################################
X_PATH = r"E:\ukbprotein\Data\Multiclass\tx_500_may.csv" 
Y_PATH = r"E:\ukbprotein\Data\Multiclass\ty_may.csv" 
OUT_DIR = r"E:\ukbprotein\Data\Multiclass\500_may"       
############################################

# ============ 超参 ============
SEED = 42
TEST_RATIO = 0.20            # 测试集占比（分层）
VAL_RATIO  = 0.10            # 验证集占比（分层；从剩余训练中切出）
BATCH_SIZE = 512
EPOCHS_MAX = 300             # 最多训练轮数（早停会提前结束）
LR = 2e-3                    # 稍大一点的起始学习率
WEIGHT_DECAY = 5e-4          # 稍弱一点的 L2 正则
DROPOUT_P = 0.3              # 放松 Dropout
LABEL_SMOOTHING = 0.0        # 先关掉（等学起来再考虑加 0.05）
INPUT_NOISE_STD = 0.0        # 先关掉（之前 0.01 容易欠拟合）
PATIENCE = 20                # 早停耐心（看验证集）

# ============ 环境 ============
os.makedirs(OUT_DIR, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============ 读数据 ============
X = np.loadtxt(X_PATH, delimiter=",", dtype=float)   # (N, 1463)
y = np.loadtxt(Y_PATH, delimiter=",", dtype=int).reshape(-1)  # (N,)
# 如果上一行报错，你的 numpy 版本没有 loadxt：请改回 loadtxt
# y = np.loadtxt(Y_PATH, delimiter=",", dtype=int).reshape(-1)

assert X.ndim == 2 and y.ndim == 1 and X.shape[0] == y.shape[0], "数据形状不匹配"
N, INPUT_DIM = X.shape
NUM_CLASSES = int(y.max()) + 1
assert NUM_CLASSES == 13, f"检测到标签范围 0..{NUM_CLASSES-1}，应为 13 类(0..12)"

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# ============ 分层划分 ============
def stratified_split_indices(y_np, ratio, seed):
    """返回(train_idx, heldout_idx)，heldout 占比为 ratio，分层抽样"""
    rng = np.random.default_rng(seed)
    y_np = np.asarray(y_np)
    idx_train, idx_held = [], []
    for cls in np.unique(y_np):
        idx = np.where(y_np == cls)[0]
        rng.shuffle(idx)
        n_held = max(1, int(len(idx) * ratio))
        idx_held.append(idx[:n_held])
        idx_train.append(idx[n_held:])
    return np.concatenate(idx_train), np.concatenate(idx_held)

# 先拿出测试集 20%
idx_remain, idx_test = stratified_split_indices(y_tensor.numpy(), TEST_RATIO, SEED)

# 从剩余中再切出验证集 10%（相对于全量约 10%）
val_ratio_in_remain = VAL_RATIO / (1.0 - TEST_RATIO)
y_remain = y_tensor[idx_remain].numpy()
idx_train_rel, idx_val_rel = stratified_split_indices(y_remain, val_ratio_in_remain, SEED+1)
idx_train = idx_remain[idx_train_rel]
idx_val   = idx_remain[idx_val_rel]

train_ds = Subset(TensorDataset(X_tensor, y_tensor), idx_train)
val_ds   = Subset(TensorDataset(X_tensor, y_tensor), idx_val)
test_ds  = Subset(TensorDataset(X_tensor, y_tensor), idx_test)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                      num_workers=0, pin_memory=torch.cuda.is_available())
val_dl   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=0, pin_memory=torch.cuda.is_available())
test_dl  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=0, pin_memory=torch.cuda.is_available())

# 统计类别分布（训练/验证/测试）
def binc(y_np): return np.bincount(y_np, minlength=NUM_CLASSES).astype(int).tolist()
print("Train class counts:", binc(y_tensor[idx_train].numpy()))
print("Valid class counts:", binc(y_tensor[idx_val].numpy()))
print("Test  class counts:", binc(y_tensor[idx_test].numpy()))
print(f"Random baseline ~ 1/13 = {1.0/13:.4f}")

# ============ 模型 ============
class WideMLP(nn.Module):
    def __init__(self, in_dim, num_classes, p_drop=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),

            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.net(x)

model = WideMLP(INPUT_DIM, NUM_CLASSES, p_drop=DROPOUT_P).to(device)

# 类别权重（用训练集频率倒数，均值归一化）
train_counts = np.bincount(y_tensor[idx_train].numpy(), minlength=NUM_CLASSES).astype(np.float32)
w = 1.0 / np.clip(train_counts, 1.0, None)
w = w / w.mean()
class_weights = torch.tensor(w, dtype=torch.float32, device=device)

# 交叉熵（若你的 PyTorch 不支持 label_smoothing 参数会抛 TypeError）
try:
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
except TypeError:
    criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=5  # 旧版 PyTorch 无 verbose 参数
)

# ============ 评估函数 ============
@torch.no_grad()
def accuracy(dataloader):
    model.eval()
    correct, total = 0, 0
    for xb, yb in dataloader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb).argmax(1)
        correct += (pred == yb).sum().item()
        total   += yb.size(0)
    return correct / max(total, 1)

# ============ 训练（早停看验证集） ============
best_val, wait = 0.0, 0
best_state = None

for epoch in range(1, EPOCHS_MAX + 1):
    model.train()
    total_loss, total_num = 0.0, 0
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)

        # 可选：输入噪声（当前关闭）
        if INPUT_NOISE_STD > 0:
            xb = xb + INPUT_NOISE_STD * torch.randn_like(xb)

        logits = model(xb)
        loss = criterion(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        total_num  += xb.size(0)

    train_loss = total_loss / max(total_num, 1)
    val_acc = accuracy(val_dl)
    print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | val_acc={val_acc:.4f}")

    scheduler.step(val_acc)  # 根据验证准确率调学习率

    # 早停：看验证集
    if val_acc > best_val + 1e-4:
        best_val = val_acc
        wait = 0
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    else:
        wait += 1
        if wait >= PATIENCE:
            print("Early stopping (on validation) triggered.")
            break

# 恢复验证集最优权重
if best_state is not None:
    model.load_state_dict(best_state)

# ============ 测试集评估与导出 ============
# ============ 对全量数据推理并保存 ============
@torch.no_grad()
def infer_all(X_tensor, batch_size=512):
    model.eval()
    preds, probs = [], []
    for i in range(0, len(X_tensor), batch_size):
        xb = X_tensor[i:i+batch_size].to(device)
        logits = model(xb)
        p = torch.softmax(logits, dim=1).cpu().numpy()
        yhat = logits.argmax(1).cpu().numpy()
        preds.append(yhat)
        probs.append(p)
    return np.concatenate(preds), np.vstack(probs)

yhat_all, proba_all = infer_all(X_tensor)

# 保存所有样本的预测结果
np.savetxt(os.path.join(OUT_DIR, "y_all.csv"), y_tensor.numpy(), fmt="%d", delimiter=",")
np.savetxt(os.path.join(OUT_DIR, "yhat_all.csv"), yhat_all, fmt="%d", delimiter=",")
np.savetxt(os.path.join(OUT_DIR, "proba_all.csv"), proba_all, fmt="%.6f", delimiter=",")
print(f"Saved all-sample predictions under: {OUT_DIR}")


