#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# LOSO Cross Validation
# θ band (4–7 Hz) / 200–700 ms

import numpy as np
import csv
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import filedialog

# =========================
# 設定
# =========================
N_SUBJECTS = 4
CHANNEL = "FCz"
FS = 1000
PRE_MS  = 1000
POST_MS = 2000
EPOCH_LEN = PRE_MS + POST_MS + 1

W = 6.0
FREQ = np.linspace(1, 100, 100)
WIDTHS = W * FS / (2 * np.pi * FREQ)

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
# データ抽出
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
# θ帯域
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

# =========================
# GUI
# =========================
root = tk.Tk()
root.withdraw()

subjects_data = []

for subj in range(1, N_SUBJECTS+1):
    print(f"\n=== Subject {subj} ===")
    ttlfile  = filedialog.askopenfilename(title=f"Sub{subj} TimeEvent CSV")
    eegfile  = filedialog.askopenfilename(title=f"Sub{subj} EEG file")
    errpfile = filedialog.askopenfilename(title=f"Sub{subj} Direction_data.csv")

    Err, Cor = extract_subject_fcz(ttlfile, eegfile, errpfile)
    Err_tf = make_theta_dataset(Err)
    Cor_tf = make_theta_dataset(Cor)
    
    len_err = len(Err_tf)
    len_cor = len(Cor_tf)

    len_min = min(len_err, len_cor)

    # ランダム抽出
    err_idx = np.random.choice(len_err, len_min, replace=False)
    cor_idx = np.random.choice(len_cor, len_min, replace=False)

    Err_tf_bal = Err_tf[err_idx]
    Cor_tf_bal = Cor_tf[cor_idx]

    X = np.vstack([Err_tf_bal, Cor_tf_bal])
    y = np.hstack([
        np.ones(len_min),
        np.zeros(len_min)
    ])
    subjects_data.append((X, y))

# =========================
# モデル定義
# =========================
class ThetaModel(nn.Module):
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
        x = torch.sigmoid(self.fc2(x))
        return x.squeeze()

# =========================
# LOSO
# =========================
fold_acc = []

for test_idx in range(N_SUBJECTS):

    print(f"\n=== Fold {test_idx+1} (Test: Sub{test_idx+1}) ===")

    X_train = []
    y_train = []
    X_test, y_test = subjects_data[test_idx]

    for i in range(N_SUBJECTS):
        if i != test_idx:
            X_train.append(subjects_data[i][0])
            y_train.append(subjects_data[i][1])

    X_train = np.vstack(X_train)
    y_train = np.hstack(y_train)

    X_train = torch.tensor(X_train[:,None,:,:], dtype=torch.float32)
    X_test  = torch.tensor(X_test[:,None,:,:], dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test  = torch.tensor(y_test, dtype=torch.float32)

    model = ThetaModel()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 学習
    for epoch in range(20):
        optimizer.zero_grad()
        out = model(X_train)
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        pred_raw = model(X_test)
        print(pred_raw[:10])

    # 評価
    with torch.no_grad():
        pred = model(X_test)
        pred = (pred > 0.5).float() #確率50%以上
        acc = accuracy_score(y_test.numpy(), pred.numpy())

    print("Accuracy:", acc)
    fold_acc.append(acc)

print("\n===== RESULT =====")
print("Mean Accuracy:", np.mean(fold_acc))
print("Std:", np.std(fold_acc))