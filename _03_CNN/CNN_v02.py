
# Change Log
# 2025-7-29: im2col + GEMM 최적화

import cupy as cp
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

from MLPj4 import MLP

def ReLU(x):
  return cp.maximum(0, x)

def ReLU_derivative(x):
  return cp.where(x > 0, 1, 0)

# --- 유틸리티 함수: im2col / col2im --- #

def get_im2col_indices(x_shape, FH, FW, pad, stride):
  N, C, H, W = x_shape
  H_out = (H + 2 * pad - FH) // stride + 1
  W_out = (W + 2 * pad - FW) // stride + 1
  i0 = cp.repeat(cp.arange(FH), FW)
  i0 = cp.tile(i0, C)
  i1 = stride * cp.repeat(cp.arange(H_out), W_out)
  j0 = cp.tile(cp.arange(FW), FH * C)
  j1 = stride * cp.tile(cp.arange(W_out), H_out)
  i = i0.reshape(-1, 1) + i1.reshape(1, -1)
  j = j0.reshape(-1, 1) + j1.reshape(1, -1)
  k = cp.repeat(cp.arange(C), FH * FW).reshape(-1, 1)
  return (k, i, j)

def im2col_gpu(x, FH, FW, pad, stride):
  x_padded = cp.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
  k, i, j = get_im2col_indices(x.shape, FH, FW, pad, stride)
  cols = x_padded[:, k, i, j]
  C = x.shape[1]
  cols = cols.transpose(1, 2, 0).reshape(FH * FW * C, -1)
  return cols

def col2im_gpu(cols, x_shape, FH, FW, pad, stride):
  N, C, H, W = x_shape
  H_padded, W_padded = H + 2 * pad, W + 2 * pad
  x_padded = cp.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
  k, i, j = get_im2col_indices(x_shape, FH, FW, pad, stride)
  cols_reshaped = cols.reshape(C * FH * FW, -1, N)
  cols_reshaped = cols_reshaped.transpose(2, 0, 1)
  cp.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
  return x_padded[:, :, pad:pad + H, pad:pad + W]

class CNN:
  def __init__(self, input_shape=(28, 28, 1), num_filters=[8, 16], filter_size=(3, 3), stride=1, pad=1,
               pool_size=2, pool_stride=2, mlp_layer_sizes=[128, 10], activation_func=ReLU, activation_func_derivative=ReLU_derivative):
    self.input_H, self.input_W, self.input_C = input_shape
    self.num_conv_layers = len(num_filters)
    self.conv_params = []
    self.pool_params = {'size': pool_size, 'stride': pool_stride}
    self.activation_func = activation_func
    self.activation_func_derivative = activation_func_derivative

    C_in = self.input_C
    H, W = self.input_H, self.input_W
    for i in range(self.num_conv_layers):
      C_out = num_filters[i]
      FH, FW = filter_size
      filters = cp.random.randn(C_out, C_in, FH, FW) * cp.sqrt(2. / (C_in * FH * FW))
      bias = cp.zeros(C_out)
      self.conv_params.append({'filters': filters, 'bias': bias, 'stride': stride, 'pad': pad})
      C_in = C_out
      H = (H - FH + 2 * pad) // stride + 1
      W = (W - FW + 2 * pad) // stride + 1
      H = (H - pool_size) // pool_stride + 1
      W = (W - pool_size) // pool_stride + 1

    flattened_size = H * W * C_out
    self.mlp = MLP([flattened_size] + mlp_layer_sizes)

  def _convolution_forward(self, A_prev, W, b, stride, pad):
    """im2col + GEMM 기반 고속 순전파"""
    # Cupy는 NCHW 형식을 더 효율적으로 처리
    A_prev_nchw = A_prev.transpose(0, 3, 1, 2)
    N, C, H, W_shape = A_prev_nchw.shape
    F, _, FH, FW = W.shape
    
    H_out = (H - FH + 2 * pad) // stride + 1
    W_out = (W_shape - FW + 2 * pad) // stride + 1

    col = im2col_gpu(A_prev_nchw, FH, FW, pad, stride)
    W_row = W.reshape(F, -1)
    
    out = W_row @ col + b.reshape(-1, 1)
    out = out.reshape(F, H_out, W_out, N)
    out = out.transpose(3, 1, 2, 0) # NHWC 형식으로 복원
    
    cache = (A_prev, W, b, stride, pad, col)
    return out, cache

  def _pooling_forward(self, A_prev, size, stride):
    """느린 for 루프 기반 풀링 순전파 (CNNj1과 동일)"""
    N, H_prev, W_prev, C_prev = A_prev.shape
    H_out = (H_prev - size) // stride + 1
    W_out = (W_prev - size) // stride + 1
    A = cp.zeros((N, H_out, W_out, C_prev))
    for n in range(N):
      for h in range(H_out):
        for w in range(W_out):
          for c in range(C_prev):
            h_start, w_start = h * stride, w * stride
            h_end, w_end = h_start + size, w_start + size
            A_slice = A_prev[n, h_start:h_end, w_start:w_end, c]
            A[n, h, w, c] = cp.max(A_slice)
    cache = (A_prev, size, stride)
    return A, cache

  def forward(self, x):
    caches = []
    A = x
    for i in range(self.num_conv_layers):
      params = self.conv_params[i]
      Z, conv_cache = self._convolution_forward(A, params['filters'], params['bias'], params['stride'], params['pad'])
      caches.append(conv_cache)
      A = self.activation_func(Z)
      relu_cache = Z
      caches.append(relu_cache)
      A, pool_cache = self._pooling_forward(A, self.pool_params['size'], self.pool_params['stride'])
      caches.append(pool_cache)
    N, H, W, C = A.shape
    A_flat = A.reshape(N, -1)
    caches.append(A.shape)
    mlp_A, mlp_Z = self.mlp.front_propagation(A_flat)
    caches.append((mlp_A, mlp_Z))
    return mlp_A[-1], caches

  def _convolution_backward(self, dZ, cache):
    """im2col + GEMM 기반 고속 역전파"""
    A_prev, W, b, stride, pad, col = cache
    N, H, W_shape, C = A_prev.shape
    F, _, FH, FW = W.shape

    db = cp.sum(dZ, axis=(0, 1, 2))
    
    dZ_reshaped = dZ.transpose(3, 1, 2, 0).reshape(F, -1)
    dW = (dZ_reshaped @ col.T).reshape(W.shape)
    
    W_row = W.reshape(F, -1)
    dout_col = W_row.T @ dZ_reshaped
    
    # NCHW 형식으로 변환하여 col2im 수행
    A_prev_nchw = A_prev.transpose(0, 3, 1, 2)
    dA_prev_nchw = col2im_gpu(dout_col, A_prev_nchw.shape, FH, FW, pad, stride)
    # NHWC 형식으로 복원
    dA_prev = dA_prev_nchw.transpose(0, 2, 3, 1)
    
    return dA_prev, dW, db

  def _pooling_backward(self, dA, cache):
    """느린 for 루프 기반 풀링 역전파 (CNNj1과 동일)"""
    A_prev, size, stride = cache
    N, H_prev, W_prev, C_prev = A_prev.shape
    _, H_out, W_out, _ = dA.shape
    dA_prev = cp.zeros_like(A_prev)
    for n in range(N):
      for h in range(H_out):
        for w in range(W_out):
          for c in range(C_prev):
            h_start, w_start = h * stride, w * stride
            h_end, w_end = h_start + size, w_start + size
            A_slice = A_prev[n, h_start:h_end, w_start:w_end, c]
            mask = (A_slice == cp.max(A_slice))
            dA_prev[n, h_start:h_end, w_start:w_end, c] += mask * dA[n, h, w, c]
    return dA_prev

  def backward(self, d_pred, caches, lr):
    mlp_A, mlp_Z = caches.pop()
    d_flat = self.mlp.backward(d_pred, mlp_A, mlp_Z, lr, d_pred.shape[0])
    prev_shape = caches.pop()
    dA = d_flat.reshape(prev_shape)
    for i in reversed(range(self.num_conv_layers)):
      pool_cache = caches.pop()
      dA = self._pooling_backward(dA, pool_cache)
      relu_cache = caches.pop()
      dZ = dA * self.activation_func_derivative(relu_cache)
      conv_cache = caches.pop()
      dA, dW, db = self._convolution_backward(dZ, conv_cache)
      self.conv_params[i]['filters'] -= lr * dW
      self.conv_params[i]['bias'] -= lr * db

  def train(self, x, y, lr, epochs, batch_size=32):
    num_samples = x.shape[0]
    loss_history = []
    for epoch in range(epochs):
      permutation = cp.random.permutation(num_samples)
      x_shuffled = x[permutation]
      y_shuffled = y[permutation]
      epoch_loss = 0
      num_batches = 0
      for i in range(0, num_samples, batch_size):
        x_batch = x_shuffled[i:i+batch_size]
        y_batch = y_shuffled[i:i+batch_size]
        pred, caches = self.forward(x_batch)
        pred = cp.clip(pred, 1e-9, 1 - 1e-9)
        loss = -cp.mean(cp.sum(y_batch * cp.log(pred), axis=1))
        d_pred = pred - y_batch
        self.backward(d_pred, caches, lr)
        epoch_loss += loss.item()
        num_batches += 1
      avg_loss = epoch_loss / num_batches
      loss_history.append(avg_loss)
      print(f"Epoch {epoch + 1}/{epochs}, Avg Loss: {avg_loss:.6f}")
    return loss_history

  def predict(self, x):
    pred, _ = self.forward(x)
    return pred

if __name__ == "__main__":
  print("Module Imported: Refactored CNN v2 (im2col+GEMM)")
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  print("MNIST Data Loaded")

  x_train = cp.asarray(x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0)
  x_test = cp.asarray(x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0)
  y_train_onehot = cp.eye(10)[y_train]
  print("Data Preprocessing Completed")

  cnn = CNN(input_shape=(28, 28, 1), num_filters=[8, 16], filter_size=(3, 3), stride=1, pad=1,
            mlp_layer_sizes=[128, 10])
  print("Model Instantiated")

  train_size = 5000
  x_train_sample = x_train[:train_size]
  y_train_sample = y_train_onehot[:train_size]

  print("Training started...")
  loss_history = cnn.train(x_train_sample, y_train_sample, lr=0.01, epochs=5, batch_size=64)
  print("Training Completed")

  print("Evaluating accuracy...")
  test_size = 1000
  predictions = cnn.predict(x_test[:test_size])
  predicted_labels = cp.argmax(predictions, axis=1)
  true_labels = y_test[:test_size]
  accuracy = cp.mean(predicted_labels == cp.asarray(true_labels))
  print(f"Accuracy on {test_size} test samples: {accuracy.get() * 100:.2f}%")

  plt.plot(loss_history)
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.title("Training Loss History (CNN v2)")
  plt.grid(True)
  plt.show()