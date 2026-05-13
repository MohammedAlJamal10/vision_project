"""
models/cnn/layers.py
====================
Differentiable layer primitives for the scratch CNN.

Each layer is a class with:
    forward(x)  → output, stored cache for backward
    backward(dout) → gradient w.r.t. input (and update W/b grads)

Shapes follow (N, C, H, W) convention (channels-first).
"""

import numpy as np


# ── Convolution ───────────────────────────────────────────────────────────────

class Conv2D:
    """2-D convolution layer with multi-channel support.

    Parameters
    ----------
    in_channels, out_channels : int
    kernel_size : int  (square kernel)
    stride      : int
    padding     : int  (zero-padding on each side)
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1, padding: int = 0):
        self.in_ch   = in_channels
        self.out_ch  = out_channels
        self.ksize   = kernel_size
        self.stride  = stride
        self.padding = padding

        # He initialisation
        fan_in = in_channels * kernel_size * kernel_size
        scale  = np.sqrt(2.0 / fan_in)
        rng    = np.random.default_rng(0)
        self.W  = rng.normal(0, scale, (out_channels, in_channels, kernel_size, kernel_size))
        self.b  = np.zeros(out_channels, dtype=np.float64)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._cache = None

    def _pad(self, x: np.ndarray) -> np.ndarray:
        if self.padding == 0:
            return x
        return np.pad(x, ((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding)))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (N, C_in, H, W)  →  (N, C_out, H_out, W_out)"""
        N, C, H, W = x.shape
        p, s, k = self.padding, self.stride, self.ksize
        H_out = (H + 2*p - k)//s + 1
        W_out = (W + 2*p - k)//s + 1

        xp  = self._pad(x)
        out = np.zeros((N, self.out_ch, H_out, W_out), dtype=np.float64)

        # im2col trick for vectorised convolution
        col = self._im2col(xp, k, s)            # (N, C*k*k, H_out*W_out)
        W_r = self.W.reshape(self.out_ch, -1)    # (out_ch, C*k*k)
        out_r = W_r @ col                        # (N, out_ch, H_out*W_out)
        out = out_r.reshape(N, self.out_ch, H_out, W_out)
        out += self.b[np.newaxis, :, np.newaxis, np.newaxis]

        self._cache = (x, xp, col)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """dout: (N, C_out, H_out, W_out)  → dx: (N, C_in, H, W)"""
        x, xp, col = self._cache
        N, C, H, W = x.shape
        k, s, p    = self.ksize, self.stride, self.padding
        H_out, W_out = dout.shape[2], dout.shape[3]

        dout_r = dout.reshape(N, self.out_ch, -1)          # (N, out_ch, H_out*W_out)
        W_r    = self.W.reshape(self.out_ch, -1)            # (out_ch, C*k*k)

        # Gradients w.r.t. weights and bias
        self.dW = np.einsum("noi,nki->ok", dout_r, col).reshape(self.W.shape) / N
        self.db = dout_r.sum(axis=(0, 2)) / N               # (out_ch,)

        # Gradient w.r.t. input
        dcol    = (W_r.T @ dout_r)                          # (N, C*k*k, H_out*W_out)
        dx      = self._col2im(dcol, x.shape, k, s, p)
        return dx

    # ── im2col / col2im ───────────────────────────────────────────────────────

    @staticmethod
    def _im2col(x_pad: np.ndarray, k: int, s: int) -> np.ndarray:
        N, C, H, W = x_pad.shape
        H_out = (H - k)//s + 1
        W_out = (W - k)//s + 1
        col   = np.zeros((N, C * k * k, H_out * W_out), dtype=np.float64)
        idx   = 0
        for i in range(H_out):
            for j in range(W_out):
                patch = x_pad[:, :, i*s:i*s+k, j*s:j*s+k]   # (N, C, k, k)
                col[:, :, idx] = patch.reshape(N, -1)
                idx += 1
        return col

    @staticmethod
    def _col2im(col: np.ndarray, x_shape: tuple,
                k: int, s: int, p: int) -> np.ndarray:
        N, C, H, W = x_shape
        H_pad = H + 2*p; W_pad = W + 2*p
        H_out = (H_pad - k)//s + 1
        W_out = (W_pad - k)//s + 1
        dx_pad = np.zeros((N, C, H_pad, W_pad), dtype=np.float64)
        idx    = 0
        for i in range(H_out):
            for j in range(W_out):
                patch = col[:, :, idx].reshape(N, C, k, k)
                dx_pad[:, :, i*s:i*s+k, j*s:j*s+k] += patch
                idx += 1
        return dx_pad[:, :, p:p+H, p:p+W] if p > 0 else dx_pad


# ── ReLU ──────────────────────────────────────────────────────────────────────

class ReLU:
    def __init__(self):
        self._mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._mask = x > 0
        return x * self._mask

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout * self._mask


# ── MaxPool ───────────────────────────────────────────────────────────────────

class MaxPool2D:
    """2-D max-pooling layer.

    Parameters
    ----------
    pool_size : int  (square pool window, default 2)
    stride    : int  (default = pool_size)
    """

    def __init__(self, pool_size: int = 2, stride: int | None = None):
        self.pool = pool_size
        self.stride = stride if stride is not None else pool_size
        self._cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (N, C, H, W) → (N, C, H', W')"""
        N, C, H, W = x.shape
        p, s = self.pool, self.stride
        H_out = (H - p)//s + 1
        W_out = (W - p)//s + 1
        out   = np.zeros((N, C, H_out, W_out), dtype=np.float64)
        mask  = np.zeros_like(x, dtype=bool)

        for i in range(H_out):
            for j in range(W_out):
                window  = x[:, :, i*s:i*s+p, j*s:j*s+p]      # (N,C,p,p)
                out[:, :, i, j] = window.reshape(N, C, -1).max(axis=-1)
                # Track argmax for backward
                flat_idx = window.reshape(N, C, -1).argmax(axis=-1)
                ri = flat_idx // p; ci = flat_idx % p
                for n in range(N):
                    for c in range(C):
                        mask[n, c, i*s+ri[n,c], j*s+ci[n,c]] = True
        self._cache = (x.shape, mask, (H_out, W_out))
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        x_shape, mask, (H_out, W_out) = self._cache
        N, C, H, W = x_shape
        p, s = self.pool, self.stride
        dx = np.zeros(x_shape, dtype=np.float64)

        for i in range(H_out):
            for j in range(W_out):
                d = dout[:, :, i, j][:, :, np.newaxis, np.newaxis]  # (N,C,1,1)
                local_mask = mask[:, :, i*s:i*s+p, j*s:j*s+p]
                dx[:, :, i*s:i*s+p, j*s:j*s+p] += d * local_mask
        return dx


# ── Flatten ───────────────────────────────────────────────────────────────────

class Flatten:
    def __init__(self):
        self._shape = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout.reshape(self._shape)


# ── Fully Connected ───────────────────────────────────────────────────────────

class Linear:
    """Fully connected (dense) layer.

    Parameters
    ----------
    in_features, out_features : int
    """

    def __init__(self, in_features: int, out_features: int):
        scale  = np.sqrt(2.0 / in_features)
        rng    = np.random.default_rng(1)
        self.W  = rng.normal(0, scale, (in_features, out_features))
        self.b  = np.zeros(out_features, dtype=np.float64)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._cache = x
        return x @ self.W + self.b

    def backward(self, dout: np.ndarray) -> np.ndarray:
        x = self._cache
        N = x.shape[0]
        self.dW = (x.T @ dout) / N
        self.db = dout.mean(axis=0)
        return dout @ self.W.T