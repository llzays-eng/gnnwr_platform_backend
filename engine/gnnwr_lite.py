"""
GNNWR / GTNNWR 的离线纯 numpy 实现（gnnwr_lite）。

设计动机
--------
GNNWR 的核心思想：回归系数不是全局常数，而是**空间位置的函数**，由一个
神经网络（原文的 SWNN 时空邻近网络）给出。等价地可写作"空间变系数模型"：

    ŷ_i = β_0(s_i) + Σ_k β_k(s_i) · x_ik

其中 β(s) 由一个小型 MLP 以坐标 s=(lon,lat) 为输入产出；GTNNWR 只需把
时间 t 并入输入 s=(lon,lat,t)，系数便同时随空间与时间变化。

本实现用手写 MLP + Adam + 早停完成端到端训练，因而能同时给出：
  · 每个观测点的一整套局部回归系数 β(s_i)      → 空间权重/系数热力图
  · 训练/验证 loss 曲线                        → 训练监控
  · 残差、R²/RMSE/MAE/AICc                      → 精度诊断与对比

与真实包的关系
--------------
生产环境应替换为导师团队开源的 `gnnwr` PyTorch 包（GPU、完整 STPNN）。
本模块是"无 GPU / 无 torch"时仍能跑通完整闭环的等价轻量版，对外接口
（fit / predict / local_coefficients / history）与平台其余部分解耦，
切换真实包时后端与看板均无需改动。
"""
from __future__ import annotations

import numpy as np

from .metrics import all_metrics


def _tanh(x):
    return np.tanh(x)


def _dtanh(a):          # a = tanh(z)
    return 1.0 - a * a


class _Standardizer:
    def fit(self, X):
        self.mu = X.mean(0)
        self.sd = X.std(0)
        self.sd[self.sd < 1e-9] = 1.0
        return self

    def transform(self, X):
        return (X - self.mu) / self.sd


class GNNWRLite:
    """
    坐标→系数 的 MLP。hidden 指隐藏层神经元数列表，如 [32, 16]。

    参数
    ----
    hidden      : 隐藏层结构（对应前端"网络结构"配置）
    lr          : 学习率
    max_epoch   : 最大轮数
    patience    : 早停容忍轮数（验证集）
    dropout     : 训练期对隐藏层的随机置零比例
    l2_coef     : 对输出系数的 L2 正则，鼓励系数空间平滑、抑制过拟合
    seed        : 随机种子
    """

    def __init__(self, hidden=(32, 16), lr=0.01, max_epoch=200, patience=25,
                 dropout=0.0, l2_coef=1e-3, batch_size=None, seed=42):
        self.hidden = list(hidden)
        self.lr = lr
        self.max_epoch = max_epoch
        self.patience = patience
        self.dropout = dropout
        self.l2_coef = l2_coef
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.history = {"train_loss": [], "val_loss": []}

    # ---------- 参数初始化 ----------
    def _init_params(self, d_in, d_out):
        sizes = [d_in] + self.hidden + [d_out]
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            fan_in = sizes[i]
            scale = np.sqrt(2.0 / fan_in)            # He 初始化
            self.W.append(self.rng.normal(0, scale, (sizes[i], sizes[i + 1])))
            self.b.append(np.zeros(sizes[i + 1]))
        # Adam 状态
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(x) for x in self.b]
        self.vb = [np.zeros_like(x) for x in self.b]
        self._t = 0

    # ---------- 前向：坐标 -> 每样本系数 ----------
    def _forward(self, C, train=False):
        acts = [C]
        drops = []
        h = C
        for i in range(len(self.W) - 1):
            z = h @ self.W[i] + self.b[i]
            a = _tanh(z)
            if train and self.dropout > 0:
                mask = (self.rng.random(a.shape) > self.dropout) / (1 - self.dropout)
                a = a * mask
                drops.append(mask)
            else:
                drops.append(None)
            acts.append(a)
            h = a
        beta = h @ self.W[-1] + self.b[-1]           # (n, p+1) 线性输出
        acts.append(beta)
        return beta, acts, drops

    # ---------- 训练 ----------
    def fit(self, coords, X, y, valid_ratio=0.15, epoch_cb=None):
        coords = np.asarray(coords, float)
        X = np.asarray(X, float)
        y = np.asarray(y, float).ravel()
        n, p = X.shape

        self.cs = _Standardizer().fit(coords)
        Cn = self.cs.transform(coords)
        Xaug = np.column_stack([np.ones(n), X])       # [1, x]，与截距系数对齐
        self._init_params(Cn.shape[1], p + 1)

        # 训练/验证划分
        idx = self.rng.permutation(n)
        n_val = max(1, int(n * valid_ratio))
        val_idx, tr_idx = idx[:n_val], idx[n_val:]
        best_val, best_state, wait = np.inf, None, 0

        bs = self.batch_size or len(tr_idx)
        for epoch in range(self.max_epoch):
            perm = self.rng.permutation(len(tr_idx))
            for st in range(0, len(tr_idx), bs):
                bidx = tr_idx[perm[st:st + bs]]
                self._step(Cn[bidx], Xaug[bidx], y[bidx])

            tr_loss = self._loss(Cn[tr_idx], Xaug[tr_idx], y[tr_idx])
            val_loss = self._loss(Cn[val_idx], Xaug[val_idx], y[val_idx])
            self.history["train_loss"].append(round(float(tr_loss), 6))
            self.history["val_loss"].append(round(float(val_loss), 6))
            if epoch_cb is not None:                  # 训练监控实时推送
                try:
                    epoch_cb({"epoch": epoch + 1,
                              "train_loss": round(float(tr_loss), 6),
                              "val_loss": round(float(val_loss), 6)})
                except Exception:
                    pass  # 回调故障不应中断训练

            if val_loss < best_val - 1e-6:
                best_val, wait = val_loss, 0
                best_state = ([w.copy() for w in self.W], [b.copy() for b in self.b])
            else:
                wait += 1
                if wait >= self.patience:
                    break

        if best_state is not None:                    # 回滚到最优
            self.W, self.b = best_state
        self._p = p
        return self

    def _loss(self, C, Xaug, y):
        beta, _, _ = self._forward(C, train=False)
        pred = np.sum(beta * Xaug, axis=1)
        mse = np.mean((pred - y) ** 2)
        reg = self.l2_coef * np.mean(beta ** 2)
        return mse + reg

    def _step(self, C, Xaug, y):
        n = len(y)
        beta, acts, drops = self._forward(C, train=True)
        pred = np.sum(beta * Xaug, axis=1)

        # 输出层梯度：dL/dbeta
        dpred = (2.0 / n) * (pred - y)                # (n,)
        gbeta = dpred[:, None] * Xaug                 # (n, p+1)
        gbeta += (2.0 * self.l2_coef / n) * beta      # L2

        grads_W = [None] * len(self.W)
        grads_b = [None] * len(self.b)
        # 反传通过各层
        delta = gbeta
        for i in reversed(range(len(self.W))):
            a_prev = acts[i]
            grads_W[i] = a_prev.T @ delta
            grads_b[i] = delta.sum(0)
            if i > 0:
                d_a = delta @ self.W[i].T
                if drops[i - 1] is not None:
                    d_a = d_a * drops[i - 1]
                delta = d_a * _dtanh(acts[i])
        self._adam_update(grads_W, grads_b)

    def _adam_update(self, gW, gb, beta1=0.9, beta2=0.999, eps=1e-8):
        self._t += 1
        lr_t = self.lr * np.sqrt(1 - beta2 ** self._t) / (1 - beta1 ** self._t)
        for i in range(len(self.W)):
            self.mW[i] = beta1 * self.mW[i] + (1 - beta1) * gW[i]
            self.vW[i] = beta2 * self.vW[i] + (1 - beta2) * (gW[i] ** 2)
            self.W[i] -= lr_t * self.mW[i] / (np.sqrt(self.vW[i]) + eps)
            self.mb[i] = beta1 * self.mb[i] + (1 - beta1) * gb[i]
            self.vb[i] = beta2 * self.vb[i] + (1 - beta2) * (gb[i] ** 2)
            self.b[i] -= lr_t * self.mb[i] / (np.sqrt(self.vb[i]) + eps)

    # ---------- 推断 ----------
    def local_coefficients(self, coords) -> np.ndarray:
        """返回每个坐标点的一整套局部系数 β(s)，形状 (n, p+1)，首列为截距。"""
        Cn = self.cs.transform(np.asarray(coords, float))
        beta, _, _ = self._forward(Cn, train=False)
        return beta

    def predict(self, coords, X) -> np.ndarray:
        beta = self.local_coefficients(coords)
        Xaug = np.column_stack([np.ones(len(X)), np.asarray(X, float)])
        return np.sum(beta * Xaug, axis=1)

    def effective_params(self) -> float:
        """
        AICc 用的有效参数近似。NN 局部模型无闭式帽子矩阵，
        这里用"(p+1) × 空间灵活度"的保守估计（灵活度随隐藏层规模温和增长），
        仅用于 AICc 量级参考，报告中已注明其为近似口径。
        """
        flex = 1.0 + np.log1p(sum(self.hidden)) / 2.0
        return float((self._p + 1) * min(flex, 4.0))

    def evaluate(self, coords, X, y) -> dict:
        pred = self.predict(coords, X)
        return all_metrics(y, pred, effective_params=self.effective_params())
