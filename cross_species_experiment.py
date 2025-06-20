import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef, f1_score, precision_score, recall_score

# 设置设备和随机种子
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
if device == "cuda":
    torch.cuda.manual_seed_all(seed)

# %%
# 1. 数据预处理函数
word_index = {'A': 0, 'T': 1, 'C': 2, 'G': 3, 'N': 4}


def process_sequence(seq, max_len=1000):
    seq = seq.upper().strip()
    if len(seq) < max_len:
        pad_len = max_len - len(seq)
        pad_positions = sorted(np.random.choice(range(1, len(seq) + 1), pad_len, replace=True))
        result = []
        pos_ptr = 0
        for i in range(max_len):
            if pos_ptr < len(pad_positions) and i == pad_positions[pos_ptr]:
                result.append('N')
                pos_ptr += 1
            if i - pos_ptr < len(seq):
                result.append(seq[i - pos_ptr])
        return ''.join(result)
    elif len(seq) > max_len:
        return seq[:max_len]
    else:
        return seq


def replace_sequence(seq):
    return [word_index.get(char, 4) for char in seq]


# %%
# 2. 加载训练数据
print("加载训练数据...")
try:
    positive_data = pd.read_csv("./data/mixed dataset/6species_a_zong1.csv")
    negative_data = pd.read_csv("./data/mixed dataset/6species_p_zong1.csv")

    print("正数据集标签分布:", positive_data["label"].value_counts())
    print("负数据集标签分布:", negative_data["label"].value_counts())

    train_data = pd.concat([positive_data, negative_data], axis=0)
    train_data = train_data.rename(columns={"sequence": "data"})
    train_data = train_data.sample(frac=1, random_state=seed).reset_index(drop=True)

    print("\n训练数据分布:")
    print("正样本数:", sum(train_data["label"] == 1))
    print("负样本数:", sum(train_data["label"] == 0))
    print("总样本数:", len(train_data))

    print(f"去重前样本数: {len(train_data)}")
    train_data = train_data.drop_duplicates(subset='data')
    print(f"去重后样本数: {len(train_data)}")

    print("\n正在预处理训练序列...")
    train_data['processed_seq'] = train_data['data'].apply(process_sequence)
    train_data['encoded_seq'] = train_data['processed_seq'].apply(replace_sequence)
    print("训练数据预处理完成!")

except Exception as e:
    print(f"训练数据加载错误: {e}")
    exit()

# 3. 加载测试数据
print("\n加载测试数据...")
try:
    test_positive = pd.read_csv("./data/cross-species/grape_a_seq.csv")
    test_negative = pd.read_csv("./data/cross-species/grape_p_seq.csv")

    # 确保测试数据有label列
    test_positive['label'] = 1
    test_negative['label'] = 0

    test_data = pd.concat([test_positive, test_negative], axis=0)
    test_data = test_data.rename(columns={"sequence": "data"})
    test_data = test_data.sample(frac=1, random_state=seed).reset_index(drop=True)

    print("\n测试数据分布:")
    print("正样本数:", sum(test_data["label"] == 1))
    print("负样本数:", sum(test_data["label"] == 0))
    print("总样本数:", len(test_data))

    print(f"测试数据去重前样本数: {len(test_data)}")
    test_data = test_data.drop_duplicates(subset='data')
    print(f"测试数据去重后样本数: {len(test_data)}")

    print("\n正在预处理测试序列...")
    test_data['processed_seq'] = test_data['data'].apply(process_sequence)
    test_data['encoded_seq'] = test_data['processed_seq'].apply(replace_sequence)
    print("测试数据预处理完成!")

except Exception as e:
    print(f"测试数据加载错误: {e}")
    exit()


# %%
# 4. 准备数据集
def make_data(X, y, max_len=1000):
    input_batch, target_batch = [], []
    for seq, label in zip(X, y):
        seq = seq[:max_len] + [4] * (max_len - len(seq)) if len(seq) < max_len else seq[:max_len]
        input_batch.append(seq)
        target_batch.append(label)
    return torch.tensor(input_batch, dtype=torch.long), torch.tensor(target_batch, dtype=torch.float32)


# 训练集和验证集划分
X_train = train_data["encoded_seq"]
y_train = train_data["label"].values.astype(np.float32).reshape(-1, 1)

# 使用10%的数据作为验证集
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=seed, stratify=y_train
)

# 测试集
X_test = test_data["encoded_seq"]
y_test = test_data["label"].values.astype(np.float32).reshape(-1, 1)

print("\n正在转换数据为Tensor...")
train_X, train_y = make_data(X_train, y_train)
val_X, val_y = make_data(X_val, y_val)
test_X, test_y = make_data(X_test, y_test)

batch_size = 128
train_dataset = Data.TensorDataset(train_X, train_y)
val_dataset = Data.TensorDataset(val_X, val_y)
test_dataset = Data.TensorDataset(test_X, test_y)

num_workers = min(4, os.cpu_count())
train_loader = Data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    num_workers=num_workers, pin_memory=True
)
val_loader = Data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False,
    num_workers=num_workers, pin_memory=True
)
test_loader = Data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False,
    num_workers=num_workers, pin_memory=True
)


# %%
# 5. 模型定义
class EfficientAdditiveAttention(nn.Module):
    def __init__(self, in_dims, token_dim, num_heads=1):
        super().__init__()
        self.to_query = nn.Linear(in_dims, token_dim * num_heads)
        self.to_key = nn.Linear(in_dims, token_dim * num_heads)
        self.w_g = nn.Parameter(torch.randn(token_dim * num_heads, 1))
        self.scale_factor = token_dim ** -0.5
        self.Proj = nn.Linear(token_dim * num_heads, token_dim * num_heads)
        self.final = nn.Linear(token_dim * num_heads, token_dim)

    def forward(self, x):
        query = self.to_query(x)
        key = self.to_key(x)
        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)
        query_weight = query @ self.w_g
        A = query_weight * self.scale_factor
        A = F.normalize(A, dim=1)
        G = torch.sum(A * query, dim=1)
        G = G.unsqueeze(1).repeat(1, key.shape[1], 1)
        out = self.Proj(G * key) + query
        return self.final(out)


class FlowAttention(nn.Module):
    def __init__(self, dim, heads=4, eps=1e-6, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.eps = eps
        self.scale = (dim // heads) ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, D = x.size()
        H = self.heads
        d = D // H

        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, N, H, d).transpose(1, 2)
        k = k.view(B, N, H, d).transpose(1, 2)
        v = v.view(B, N, H, d).transpose(1, 2)

        q = F.relu(q)
        k = F.relu(k)

        k_sum = k.sum(dim=2, keepdim=True) + self.eps

        z = 1.0 / (torch.einsum('bhnd,bhkd->bhnk', q, k_sum).squeeze(-1) + self.eps)
        z = z.unsqueeze(-1)

        kv = torch.einsum('bhnd,bhnc->bhdc', k, v)
        attention = torch.einsum('bhnd,bhdc->bhnc', q, kv)

        out = attention * z
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.to_out(self.dropout(out))


class DeepFI(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=5, embedding_dim=8, padding_idx=4)

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(8, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        )

        self.flow_attn = FlowAttention(dim=64)

        self.multiscale = nn.ModuleDict({
            'conv3': nn.Conv1d(4, 32, kernel_size=3, padding=1),
            'conv5': nn.Conv1d(4, 32, kernel_size=5, padding=2),
            'conv7': nn.Conv1d(4, 32, kernel_size=7, padding=3),
            'conv1': nn.Conv1d(4, 32, kernel_size=1)
        })

        self.lstm = nn.LSTM(64, 32, num_layers=2, bidirectional=True, dropout=0.3, batch_first=True)

        self.post_combine_attn = EfficientAdditiveAttention(in_dims=128, token_dim=128, num_heads=4)

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, X):
        X = X.long()
        x_emb = self.emb(X).permute(0, 2, 1)
        x1 = self.conv_block1(x_emb)
        x2 = self.conv_block2(x1)
        x3 = self.conv_block3(x2)

        attn_input = x3.permute(0, 2, 1)
        attn_out = self.flow_attn(attn_input)
        attn_out = attn_out.permute(0, 2, 1)

        x_onehot = F.one_hot(X, num_classes=5)[:, :, :-1].float().permute(0, 2, 1)
        ms_features = []
        for name, layer in self.multiscale.items():
            ms_features.append(F.relu(layer(x_onehot)))
        x_concat = torch.cat(ms_features, dim=1)

        lstm_input = x_concat.permute(0, 2, 1)[:, :, :64]
        lstm_out, _ = self.lstm(lstm_input)

        combined = torch.cat([attn_out.permute(0, 2, 1), lstm_out], dim=2)
        combined = self.post_combine_attn(combined)
        pooled = torch.max(combined, dim=1)[0]
        return self.classifier(pooled)


# %%
# 6. 训练和评估函数
model = DeepFI().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)


class EnhancedFGM:
    def __init__(self, model, eps=0.1):
        self.model = model
        self.eps = eps
        self.backup = {}

    def attack(self, emb_name='emb'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and (emb_name in name or 'conv' in name):
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.eps * param.grad / (norm + 1e-8)
                    param.data.add_(r_at)

    def restore(self, emb_name='emb'):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


fgm = EnhancedFGM(model)


def evaluate(loader, return_probs=False):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    total_loss = 0.0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).long()

            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
            if return_probs:
                all_probs.extend(probs.cpu().numpy())

    if return_probs:
        return all_preds, all_labels, total_loss / len(loader), np.array(all_probs)
    return all_preds, all_labels, total_loss / len(loader)


# %%
# 7. 训练循环
best_epoch = 0
best_val_accuracy = 0
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'lr': []}

print("\n开始训练...")
for epoch in range(50):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()

        fgm.attack()
        pred_adv = model(x)
        loss_adv = criterion(pred_adv, y)
        loss_adv.backward()
        fgm.restore()

        optimizer.step()

        train_loss += loss.item()
        preds = (torch.sigmoid(pred) > 0.5).long()
        total += y.size(0)
        correct += (preds == y).sum().item()

    val_preds, val_labels, val_loss = evaluate(val_loader)
    val_accuracy = accuracy_score(val_labels, val_preds)

    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_epoch = epoch
        torch.save(model.state_dict(), "cross_species_best_model.pth")
        print(f"保存最佳模型在 epoch {epoch + 1}, 准确率: {val_accuracy:.4f}")

    history['train_loss'].append(train_loss / len(train_loader))
    history['val_loss'].append(val_loss)
    history['train_acc'].append(100 * correct / total)
    history['val_acc'].append(100 * val_accuracy)
    history['lr'].append(current_lr)

    print(f"\nEpoch {epoch + 1}/50")
    print(f"Train Loss: {history['train_loss'][-1]:.4f} | Acc: {history['train_acc'][-1]:.2f}%")
    print(f"Val Loss: {history['val_loss'][-1]:.4f} | Acc: {history['val_acc'][-1]:.2f}%")
    print(f"Learning Rate: {current_lr:.2e}")
    print("-" * 50)

# %%
# 8. 测试模型
print(f"\n加载最佳模型 (epoch {best_epoch + 1})")
model.load_state_dict(torch.load("cross_species_best_model.pth"))

test_preds, test_labels, test_loss, test_probs = evaluate(test_loader, return_probs=True)
test_labels = np.array(test_labels).astype(int)
test_preds = np.array(test_preds).astype(int)

# 计算指标
cm = confusion_matrix(test_labels, test_preds)
tn, fp, fn, tp = cm.ravel()
has_two_classes = len(np.unique(test_labels)) > 1

metrics = {
    'Accuracy': accuracy_score(test_labels, test_preds),
    'Precision': precision_score(test_labels, test_preds),
    'Recall': recall_score(test_labels, test_preds),
    'F1 Score': f1_score(test_labels, test_preds),
    'MCC': matthews_corrcoef(test_labels, test_preds),
    'TP': tp,
    'FP': fp,
    'TN': tn,
    'FN': fn
}

if has_two_classes:
    metrics['AUC'] = roc_auc_score(test_labels, test_probs.flatten())
else:
    metrics['AUC'] = 'NA (only one class)'

print("\n跨物种测试性能指标:")
for name, value in metrics.items():
    if isinstance(value, float):
        print(f"{name}: {value:.4f}")
    else:
        print(f"{name}: {value}")

# 保存指标
with open('cross_species_test_metrics.txt', 'w') as f:
    for name, value in metrics.items():
        f.write(f"{name}: {value}\n")

