
# Change Log
# 2025-7-30: Full optimization with GEMM parallelization, multi-channel support, batch processing, extended Winograd, and automatic fallback
# 2025-10-8: 학습 가능한 배치 정규화, 체크포인트 기능 추가, 학습률 스케줄링, Adam 옵티마이저 적용, 모델 구조 및 학습 방식 개선

import cupy as cp
import numpy as np
import os
import json

from .._02_MLP import MLP

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
    self.bn_params = []
    self.pool_params = {'size': pool_size, 'stride': pool_stride}
    self.activation_func = activation_func
    self.activation_func_derivative = activation_func_derivative
    self.is_training = True

    # Adam Optimizer parameters
    self.adam_params = {'t': 0, 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8}
    self.m, self.v = {}, {}

    C_in = self.input_C
    H, W = self.input_H, self.input_W
    for i in range(self.num_conv_layers):
      C_out = num_filters[i]
      FH, FW = filter_size
      filters = cp.random.randn(C_out, C_in, FH, FW) * cp.sqrt(2. / (C_in * FH * FW))
      bias = cp.zeros(C_out)
      self.conv_params.append({'filters': filters, 'bias': bias, 'stride': stride, 'pad': pad})
      
      gamma = cp.ones(C_out)
      beta = cp.zeros(C_out)
      running_mean = cp.zeros(C_out)
      running_var = cp.zeros(C_out)
      self.bn_params.append({'gamma': gamma, 'beta': beta, 'running_mean': running_mean, 'running_var': running_var, 'momentum': 0.9})
      
      # Initialize Adam moments
      self.m[f'W{i}'] = cp.zeros_like(filters)
      self.v[f'W{i}'] = cp.zeros_like(filters)
      self.m[f'b{i}'] = cp.zeros_like(bias)
      self.v[f'b{i}'] = cp.zeros_like(bias)
      self.m[f'gamma{i}'] = cp.zeros_like(gamma)
      self.v[f'gamma{i}'] = cp.zeros_like(gamma)
      self.m[f'beta{i}'] = cp.zeros_like(beta)
      self.v[f'beta{i}'] = cp.zeros_like(beta)

      C_in = C_out
      H = (H - FH + 2 * pad) // stride + 1
      W = (W - FW + 2 * pad) // stride + 1
      H = (H - pool_size) // pool_stride + 1
      W = (W - pool_size) // pool_stride + 1

    flattened_size = H * W * C_out
    self.mlp = MLP([flattened_size] + mlp_layer_sizes)

  def _convolution_forward(self, A_prev, W, b, stride, pad):
    A_prev_nchw = A_prev.transpose(0, 3, 1, 2)
    N, C, H, W_shape = A_prev_nchw.shape
    F, _, FH, FW = W.shape
    H_out = (H - FH + 2 * pad) // stride + 1
    W_out = (W_shape - FW + 2 * pad) // stride + 1
    col = im2col_gpu(A_prev_nchw, FH, FW, pad, stride)
    W_row = W.reshape(F, -1)
    out = W_row @ col + b.reshape(-1, 1)
    out = out.reshape(F, H_out, W_out, N)
    out = out.transpose(3, 1, 2, 0)
    cache = (A_prev, W, b, stride, pad, col)
    return out, cache

  def _batchnorm_forward(self, x, gamma, beta, running_mean, running_var, momentum):
    if self.is_training:
      mean = cp.mean(x, axis=(0, 1, 2))
      var = cp.var(x, axis=(0, 1, 2))
      running_mean = momentum * running_mean + (1 - momentum) * mean
      running_var = momentum * running_var + (1 - momentum) * var
    else:
      mean = running_mean
      var = running_var
    x_norm = (x - mean) / cp.sqrt(var + 1e-5)
    out = gamma * x_norm + beta
    cache = (x, x_norm, mean, var, gamma, beta)
    return out, cache, running_mean, running_var

  def _pooling_forward(self, A_prev, size, stride):
    N, H_prev, W_prev, C_prev = A_prev.shape
    H_out = (H_prev - size) // stride + 1
    W_out = (W_prev - size) // stride + 1
    A_prev_reshaped = A_prev.reshape(N * C_prev, 1, H_prev, W_prev)
    A_prev_col = im2col_gpu(A_prev_reshaped, size, size, 0, stride)
    A_col = A_prev_col.max(axis=0)
    A = A_col.reshape(H_out, W_out, N, C_prev).transpose(2, 0, 1, 3)
    cache = (A_prev, A_prev_col, size, stride)
    return A, cache

  def forward(self, x):
    caches = []
    A = x
    for i in range(self.num_conv_layers):
      params = self.conv_params[i]
      Z, conv_cache = self._convolution_forward(A, params['filters'], params['bias'], params['stride'], params['pad'])
      caches.append(conv_cache)
      bn_p = self.bn_params[i]
      BN, bn_cache, r_mean, r_var = self._batchnorm_forward(Z, bn_p['gamma'], bn_p['beta'], bn_p['running_mean'], bn_p['running_var'], bn_p['momentum'])
      self.bn_params[i]['running_mean'] = r_mean
      self.bn_params[i]['running_var'] = r_var
      caches.append(bn_cache)
      A = self.activation_func(BN)
      relu_cache = BN
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
    A_prev, W, b, stride, pad, col = cache
    F, _, FH, FW = W.shape
    db = cp.sum(dZ, axis=(0, 1, 2))
    dZ_reshaped = dZ.transpose(3, 1, 2, 0).reshape(F, -1)
    dW = (dZ_reshaped @ col.T).reshape(W.shape)
    W_row = W.reshape(F, -1)
    dout_col = W_row.T @ dZ_reshaped
    A_prev_nchw = A_prev.transpose(0, 3, 1, 2)
    dA_prev_nchw = col2im_gpu(dout_col, A_prev_nchw.shape, FH, FW, pad, stride)
    dA_prev = dA_prev_nchw.transpose(0, 2, 3, 1)
    return dA_prev, dW, db

  def _batchnorm_backward(self, d_out, cache):
    x, x_norm, mean, var, gamma, beta = cache
    N, H, W, C = d_out.shape
    dgamma = cp.sum(d_out * x_norm, axis=(0, 1, 2))
    dbeta = cp.sum(d_out, axis=(0, 1, 2))
    dx_norm = d_out * gamma
    dx = (1. / (N * H * W * cp.sqrt(var + 1e-5))) * \
         (N * H * W * dx_norm - cp.sum(dx_norm, axis=(0, 1, 2)) - x_norm * cp.sum(dx_norm * x_norm, axis=(0, 1, 2)))
    return dx, dgamma, dbeta

  def _pooling_backward(self, dA, cache):
    A_prev, A_prev_col, size, stride = cache
    N, H_prev, W_prev, C_prev = A_prev.shape
    dA_reshaped = dA.transpose(2, 3, 0, 1).ravel()
    max_idx = A_prev_col.argmax(axis=0)
    dA_col = cp.zeros_like(A_prev_col)
    dA_col[max_idx, cp.arange(max_idx.size)] = dA_reshaped
    dA_prev_reshaped = col2im_gpu(dA_col, (N * C_prev, 1, H_prev, W_prev), size, size, 0, stride)
    dA_prev = dA_prev_reshaped.reshape(A_prev.shape)
    return dA_prev

  def _update_params_adam(self, param, grad, m_key, v_key, lr):
      beta1, beta2, epsilon = self.adam_params['beta1'], self.adam_params['beta2'], self.adam_params['epsilon']
      self.m[m_key] = beta1 * self.m[m_key] + (1 - beta1) * grad
      self.v[v_key] = beta2 * self.v[v_key] + (1 - beta2) * (grad ** 2)
      m_hat = self.m[m_key] / (1 - beta1 ** self.adam_params['t'])
      v_hat = self.v[v_key] / (1 - beta2 ** self.adam_params['t'])
      param -= lr * m_hat / (cp.sqrt(v_hat) + epsilon)

  def backward(self, d_pred, caches, lr):
    self.adam_params['t'] += 1
    mlp_A, mlp_Z = caches.pop()
    d_flat = self.mlp.backward(d_pred, mlp_A, mlp_Z, lr, d_pred.shape[0])
    prev_shape = caches.pop()
    dA = d_flat.reshape(prev_shape)
    for i in reversed(range(self.num_conv_layers)):
      pool_cache = caches.pop()
      dA = self._pooling_backward(dA, pool_cache)
      relu_cache = caches.pop()
      dBN = dA * self.activation_func_derivative(relu_cache)
      bn_cache = caches.pop()
      dZ, dgamma, dbeta = self._batchnorm_backward(dBN, bn_cache)
      conv_cache = caches.pop()
      dA, dW, db = self._convolution_backward(dZ, conv_cache)
      
      # Update parameters using Adam
      self._update_params_adam(self.conv_params[i]['filters'], dW, f'W{i}', f'W{i}', lr)
      self._update_params_adam(self.conv_params[i]['bias'], db, f'b{i}', f'b{i}', lr)
      self._update_params_adam(self.bn_params[i]['gamma'], dgamma, f'gamma{i}', f'gamma{i}', lr)
      self._update_params_adam(self.bn_params[i]['beta'], dbeta, f'beta{i}', f'beta{i}', lr)

  def train(self, x, y, lr, epochs, batch_size=32, target_loss=None, verbose=True, save_every=5, checkpoint_dir="checkpoints", model_name="model", start_epoch=0):
    self.is_training = True
    num_samples = x.shape[0]
    loss_history = []
    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    current_lr = lr
    for epoch in range(start_epoch, start_epoch + epochs):
      # Learning Rate Scheduling: Halve LR every 5 epochs
      if epoch > start_epoch and (epoch - start_epoch) % 5 == 0:
          current_lr /= 2
          if verbose:
              print(f"Learning rate decayed to {current_lr:.6f}")

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
        self.backward(d_pred, caches, current_lr) # Use scheduled lr
        epoch_loss += loss.item()
        num_batches += 1
      avg_loss = epoch_loss / num_batches
      loss_history.append(avg_loss)
      
      if verbose:
        print(f"Epoch {epoch + 1}/{start_epoch + epochs}, LR: {current_lr:.6f}, Avg Loss: {avg_loss:.6f}")

      if (epoch + 1) % save_every == 0:
        self.save_model(os.path.join(checkpoint_dir, f"{model_name}_epoch_{epoch+1}.npz"), epoch=epoch + 1)

      if target_loss is not None and avg_loss <= target_loss:
        if verbose:
          print(f"Target loss {target_loss} reached. Early stopping at epoch {epoch + 1}.")
        break
    return loss_history, start_epoch + len(loss_history)

  def predict(self, x, batch_size=100):
    self.is_training = False
    num_samples = x.shape[0]
    all_preds = []
    for i in range(0, num_samples, batch_size):
        x_batch = x[i:i+batch_size]
        pred, _ = self.forward(x_batch)
        all_preds.append(pred)
    return cp.concatenate(all_preds, axis=0)

  def save_model(self, path="model.npz", epoch=0):
    params = {'num_conv_layers': self.num_conv_layers, 'epoch': epoch, 'adam_t': self.adam_params['t']}
    for i in range(self.num_conv_layers):
      params[f'conv_w_{i}'] = self.conv_params[i]['filters'].get()
      params[f'conv_b_{i}'] = self.conv_params[i]['bias'].get()
      params[f'bn_gamma_{i}'] = self.bn_params[i]['gamma'].get()
      params[f'bn_beta_{i}'] = self.bn_params[i]['beta'].get()
      params[f'bn_rmean_{i}'] = self.bn_params[i]['running_mean'].get()
      params[f'bn_rvar_{i}'] = self.bn_params[i]['running_var'].get()
      # Save Adam moments
      params[f'm_W{i}'] = self.m[f'W{i}'].get()
      params[f'v_W{i}'] = self.v[f'W{i}'].get()
      params[f'm_b{i}'] = self.m[f'b{i}'].get()
      params[f'v_b{i}'] = self.v[f'b{i}'].get()
      params[f'm_gamma{i}'] = self.m[f'gamma{i}'].get()
      params[f'v_gamma{i}'] = self.v[f'gamma{i}'].get()
      params[f'm_beta{i}'] = self.m[f'beta{i}'].get()
      params[f'v_beta{i}'] = self.v[f'beta{i}'].get()

    mlp_params = self.mlp.get_parameters()
    for i, (w, b) in enumerate(mlp_params):
      params[f"mlp_w_{i}"] = w.get()
      params[f"mlp_b_{i}"] = b.get()
    np.savez(path, **params)
    print(f"Model saved to {path}")

  def load_model(self, path):
    data = np.load(path)
    num_layers = int(data['num_conv_layers'])
    assert num_layers == self.num_conv_layers, "Model architecture mismatch!"
    self.adam_params['t'] = int(data['adam_t'].item()) if 'adam_t' in data else 0

    for i in range(num_layers):
      self.conv_params[i]['filters'] = cp.asarray(data[f'conv_w_{i}'])
      self.conv_params[i]['bias'] = cp.asarray(data[f'conv_b_{i}'])
      self.bn_params[i]['gamma'] = cp.asarray(data[f'bn_gamma_{i}'])
      self.bn_params[i]['beta'] = cp.asarray(data[f'bn_beta_{i}'])
      self.bn_params[i]['running_mean'] = cp.asarray(data[f'bn_rmean_{i}'])
      self.bn_params[i]['running_var'] = cp.asarray(data[f'bn_rvar_{i}'])
      # Load Adam moments if they exist
      if f'm_W{i}' in data:
          self.m[f'W{i}'] = cp.asarray(data[f'm_W{i}'])
          self.v[f'W{i}'] = cp.asarray(data[f'v_W{i}'])
          self.m[f'b{i}'] = cp.asarray(data[f'm_b{i}'])
          self.v[f'b{i}'] = cp.asarray(data[f'v_b{i}'])
          self.m[f'gamma{i}'] = cp.asarray(data[f'm_gamma{i}'])
          self.v[f'gamma{i}'] = cp.asarray(data[f'v_gamma{i}'])
          self.m[f'beta{i}'] = cp.asarray(data[f'm_beta{i}'])
          self.v[f'beta{i}'] = cp.asarray(data[f'v_beta{i}'])

    mlp_params = []
    i = 0
    while f"mlp_w_{i}" in data:
      w = cp.asarray(data[f"mlp_w_{i}"])
      b = cp.asarray(data[f"mlp_b_{i}"])
      mlp_params.append((w, b))
      i += 1
    self.mlp.set_parameters(mlp_params)
    
    start_epoch = int(data['epoch'].item()) if 'epoch' in data else 0
    print(f"Model loaded from {path}. Resuming from epoch {start_epoch + 1}")
    return start_epoch
