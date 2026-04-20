##CNN incremental learning
#


import numpy as np
import csv
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import filedialog
import os

# =========================
# 設定
# =========================
N_SUBJECTS = 1
CHANNEL = "FCz"
FS = 1000
PRE_MS  = 1000
POST_MS = 2000
EPOCH_LEN = PRE_MS + POST_MS + 1

W = 6.0
FREQ = np.linspace(1, 100, 100)
WIDTHS = W * FS / (2 * np.pi * FREQ)
model_path = "incremental_model.pth"

# =========================
# Morlet
# =========================
def morlet_cwt(x, widths, w=6.0):
    N = len(x)
    t = np.arange(-N/2, N/2)
    out = np.zeros((len(widths), N))
    for i, width in enumerate(widths):
        wavelet = np.exp(1j*w*t/width) * np.exp(-(t/width)**2/2)
        conv = np.convolve(x, wavelet, mode="same")
        out[i] = np.abs(conv)
    return out

# =========================
# データ抽出 p.s データ構造に合わせて修正必要
# =========================
def extract_subject_fcz(ttlfile, eegfile, errpfile):

    data = pd.read_table(eegfile, index_col=0)
    data.index = data.index.astype(float).round().astype(int)
    data.columns = ["Fz","F3","FCz","Cz","C3","C4","Pz","EOG"]

    timeevent = []
    with open(ttlfile, newline='', encoding="cp932", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("TimeEvent"):
                break
            timeevent.append(float(row["TimeEvent"]))

    CLUSTER_GAP_SEC = 1.0
    ttl_list = []
    prev_t = None
    for t in timeevent:
        if prev_t is None or (t - prev_t) >= CLUSTER_GAP_SEC:
            ttl_list.append(int(round(t*1000)))
        prev_t = t

    ttl_ms = ttl_list

    ErrP = []
    with open(errpfile, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ErrP.append(int(row["errp"]))
            except:
                continue

    ttl_ms = ttl_ms[:len(ErrP)]

    Err_trials, Cor_trials = [], []

    for i, center in enumerate(ttl_ms):
        start = center - PRE_MS
        end   = center + POST_MS
        if start < data.index[0] or end > data.index[-1]:
            continue

        epoch = data.loc[start:end]
        if len(epoch) != EPOCH_LEN:
            continue

        sig = epoch[CHANNEL].values

        if ErrP[i] == 1:
            Err_trials.append(sig)
        else:
            Cor_trials.append(sig)

    return np.array(Err_trials), np.array(Cor_trials)

# =========================
# θ帯域　p.s 全域への変更視野
# =========================
def make_theta_dataset(trials):
    freq_mask = (FREQ >= 4) & (FREQ <= 7)
    tf_list = []

    for x in trials:
        tf = morlet_cwt(x, WIDTHS, w=W)
        tf -= tf[:, 500:1000].mean(axis=1)[:, None]
        tf = tf[freq_mask, :]
        tf = tf[:, 1200:1701]
        tf_list.append(tf)

    return np.array(tf_list)

# ==========================
# subject data extraction
# ==========================
def load_subject_data(subj_id, n_train=10, n_test=10, replace=True):
    print(f"\n=== Subject {subj_id} ===")

    root = tk.Tk()
    root.withdraw()

    # ===== ファイル選択 =====
    ttlfile  = filedialog.askopenfilename(title=f"Sub{subj_id} TimeEvent CSV")
    eegfile  = filedialog.askopenfilename(title=f"Sub{subj_id} EEG file")
    errpfile = filedialog.askopenfilename(title=f"Sub{subj_id} Direction_data.csv")

    # ===== データ抽出 =====
    Err, Cor = extract_subject_fcz(ttlfile, eegfile, errpfile)
    Err_tf = make_theta_dataset(Err)
    Cor_tf = make_theta_dataset(Cor)

    # ===== データチェック =====
    if len(Err_tf) == 0 or len(Cor_tf) == 0:
        print("データ不足")
        return None

    # ===== ランダム抽出（重複あり/なし）=====
    err_train_idx = np.random.choice(len(Err_tf), n_train, replace=replace)
    err_test_idx  = np.random.choice(len(Err_tf), n_test,  replace=replace)

    cor_train_idx = np.random.choice(len(Cor_tf), n_train, replace=replace)
    cor_test_idx  = np.random.choice(len(Cor_tf), n_test,  replace=replace)

    Err_train = Err_tf[err_train_idx]
    Err_test  = Err_tf[err_test_idx]

    Cor_train = Cor_tf[cor_train_idx]
    Cor_test  = Cor_tf[cor_test_idx]

    # ===== 結合 =====
    X_train = np.vstack([Err_train, Cor_train])
    y_train = np.hstack([np.ones(n_train), np.zeros(n_train)])

    X_test = np.vstack([Err_test, Cor_test])
    y_test = np.hstack([np.ones(n_test), np.zeros(n_test)])

    x_train = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    x_test = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    return X_train, y_train, X_test, y_test
# ==========================
# model definition
# ==========================
class EEGCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(3,15), padding=(1,7))
        self.pool1 = nn.MaxPool2d((1,2))
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3,7), padding=(1,3))
        self.pool2 = nn.MaxPool2d((1,2))
        self.fc1 = nn.Linear(16*4*125, 32)
        self.fc2 = nn.Linear(32,1)

    def forward(self,x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))#ロジスティックに変更予定
        return x.squeeze()
# ==========================
# model incremental
# ==========================
model = EEGCNN()
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    lr = 0.0005   
else:
    lr = 0.001  

optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCELoss()#ロジスティックに変更予定
# ==========================
# main
# ==========================



for subj in range(1, N_SUBJECTS+1):
    x_train, y_train, x_test, y_test = load_subject_data(subj, n_train=10, n_test=10, replace=True)
    

# 学習
for epoch in range(20):
    optimizer.zero_grad()
    out = model(x_train)
    loss = criterion(out, y_train)
    loss.backward()
    optimizer.step()

with torch.no_grad():
    pred_raw = model(x_test)
    print(pred_raw[:10])
# 評価
with torch.no_grad():
    pred = model(x_test)
    pred = (pred > 0.5).float() #確率50%以上
    acc = accuracy_score(y_test.numpy(), pred.numpy())

print("Accuracy:", acc)
# 結果表示

# 結果保存

torch.save(model.state_dict(), model_path)