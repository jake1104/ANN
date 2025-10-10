
# Created at
# 2025-5-14 ~ 2025-5-18
# 2025-7-29: cupy로 변경
# 2025-10-8: 필터 및 편향 학습 적용

import cupy as cp
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# 같은 폴더의 MLPj4.py를 임포트합니다.
from MLPj4 import MLP

def ReLU(x):
  return cp.maximum(0, x)

def ReLU_derivative(x):
  return cp.where(x > 0, 1, 0)

class CNN:
  def __init__(self, input_shape=(28, 28, 1), num_filters=[8, 16], filter_size=(3, 3), stride=1, pad=1,
               pool_size=2, pool_stride=2, mlp_layer_sizes=[128, 10], activation_func=ReLU, activation_func_derivative=ReLU_derivative):
    self.input_H, self.input_W, self.input_C = input_shape
    self.num_conv_layers = len(num_filters)
    self.conv_params = []
    self.pool_params = {'size': pool_size, 'stride': pool_stride}
    self.activation_func = activation_func
    self.activation_func_derivative = activation_func_derivative

    # Conv Layer 파라미터 초기화
    C_in = self.input_C
    H, W = self.input_H, self.input_W
    for i in range(self.num_conv_layers):
      C_out = num_filters[i]
      FH, FW = filter_size
      # He 초기화
      filters = cp.random.randn(C_out, C_in, FH, FW) * cp.sqrt(2. / (C_in * FH * FW))
      # 올바른 1D 편향 벡터
      bias = cp.zeros(C_out)
      self.conv_params.append({'filters': filters, 'bias': bias, 'stride': stride, 'pad': pad})
      C_in = C_out
      # 각 레이어 통과 후 크기 계산
      H = (H - FH + 2 * pad) // stride + 1
      W = (W - FW + 2 * pad) // stride + 1
      H = (H - pool_size) // pool_stride + 1
      W = (W - pool_size) // pool_stride + 1

    # MLP Layer 초기화
    flattened_size = H * W * C_out
    self.mlp = MLP([flattened_size] + mlp_layer_sizes)

  def _convolution_forward(self, A_prev, W, b, stride, pad):
    """느린 for 루프 기반 순전파 (배치 처리)"""
    N, H_prev, W_prev, C_prev = A_prev.shape
    F, C_prev, FH, FW = W.shape
    H_out = (H_prev - FH + 2 * pad) // stride + 1
    W_out = (W_prev - FW + 2 * pad) // stride + 1
    
    Z = cp.zeros((N, H_out, W_out, F))
    A_prev_pad = cp.pad(A_prev, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant')

    for n in range(N):
      for h in range(H_out):
        for w in range(W_out):
          for f in range(F):
            h_start, w_start = h * stride, w * stride
            h_end, w_end = h_start + FH, w_start + FW
            A_slice = A_prev_pad[n, h_start:h_end, w_start:w_end, :]
            Z[n, h, w, f] = cp.sum(A_slice * W[f, :, :, :].transpose(1, 2, 0)) + b[f]
    
    cache = (A_prev, W, b, stride, pad)
    return Z, cache

  def _pooling_forward(self, A_prev, size, stride):
    """느린 for 루프 기반 풀링 순전파 (배치 처리)"""
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
    """전체 순전파 과정"""
    caches = []
    A = x
    # Conv -> ReLU -> Pool
    for i in range(self.num_conv_layers):
      params = self.conv_params[i]
      # Conv
      Z, conv_cache = self._convolution_forward(A, params['filters'], params['bias'], params['stride'], params['pad'])
      caches.append(conv_cache)
      # ReLU
      A = self.activation_func(Z)
      relu_cache = Z
      caches.append(relu_cache)
      # Pool
      A, pool_cache = self._pooling_forward(A, self.pool_params['size'], self.pool_params['stride'])
      caches.append(pool_cache)

    # Flatten
    N, H, W, C = A.shape
    A_flat = A.reshape(N, -1)
    caches.append(A.shape) # shape for unflatten

    # MLP
    mlp_A, mlp_Z = self.mlp.front_propagation(A_flat)
    caches.append((mlp_A, mlp_Z))
    
    return mlp_A[-1], caches

  def _convolution_backward(self, dZ, cache):
    """느린 for 루프 기반 역전파"""
    A_prev, W, b, stride, pad = cache
    N, H_prev, W_prev, C_prev = A_prev.shape
    F, _, FH, FW = W.shape
    _, H_out, W_out, _ = dZ.shape

    dA_prev = cp.zeros_like(A_prev)
    dW = cp.zeros_like(W)
    db = cp.zeros_like(b)

    A_prev_pad = cp.pad(A_prev, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant')
    dA_prev_pad = cp.zeros_like(A_prev_pad)

    db = cp.sum(dZ, axis=(0, 1, 2))

    for n in range(N):
      for h in range(H_out):
        for w in range(W_out):
          for f in range(F):
            h_start, w_start = h * stride, w * stride
            h_end, w_end = h_start + FH, w_start + FW
            A_slice = A_prev_pad[n, h_start:h_end, w_start:w_end, :]
            
            # dW, dA_prev 계산
            dW[f] += A_slice.transpose(2,0,1) * dZ[n, h, w, f]
            dA_prev_pad[n, h_start:h_end, w_start:w_end, :] += W[f].transpose(1,2,0) * dZ[n, h, w, f]
    
    dA_prev = dA_prev_pad[:, pad:pad+H_prev, pad:pad+W_prev, :]
    return dA_prev, dW, db

  def _pooling_backward(self, dA, cache):
    """느린 for 루프 기반 풀링 역전파"""
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
    """전체 역전파 과정"""
    # MLP
    mlp_A, mlp_Z = caches.pop()
    d_flat = self.mlp.backward(d_pred, mlp_A, mlp_Z, lr, d_pred.shape[0])

    # Unflatten
    prev_shape = caches.pop()
    dA = d_flat.reshape(prev_shape)

    # Conv/Pool Layers (역순으로)
    for i in reversed(range(self.num_conv_layers)):
      # Pool backward
      pool_cache = caches.pop()
      dA = self._pooling_backward(dA, pool_cache)
      
      # ReLU backward
      relu_cache = caches.pop()
      dZ = dA * self.activation_func_derivative(relu_cache)

      # Conv backward
      conv_cache = caches.pop()
      dA, dW, db = self._convolution_backward(dZ, conv_cache)
      
      # 파라미터 업데이트
      self.conv_params[i]['filters'] -= lr * dW
      self.conv_params[i]['bias'] -= lr * db

  def train(self, x, y, lr, epochs, batch_size=32):
    """모델 학습 메인 루프"""
    num_samples = x.shape[0]
    loss_history = []
    
    for epoch in range(epochs):
      # 데이터 셔플링
      permutation = cp.random.permutation(num_samples)
      x_shuffled = x[permutation]
      y_shuffled = y[permutation]
      
      epoch_loss = 0
      num_batches = 0

      for i in range(0, num_samples, batch_size):
        x_batch = x_shuffled[i:i+batch_size]
        y_batch = y_shuffled[i:i+batch_size]

        # 1. 순전파
        pred, caches = self.forward(x_batch)

        # 2. 손실 및 초기 그라디언트 계산
        pred = cp.clip(pred, 1e-9, 1 - 1e-9)
        loss = -cp.mean(cp.sum(y_batch * cp.log(pred), axis=1))
        d_pred = pred - y_batch

        # 3. 역전파 및 파라미터 업데이트
        self.backward(d_pred, caches, lr)
        
        epoch_loss += loss.item()
        num_batches += 1

      avg_loss = epoch_loss / num_batches
      loss_history.append(avg_loss)
      print(f"Epoch {epoch + 1}/{epochs}, Avg Loss: {avg_loss:.6f}")
    
    return loss_history

  def predict(self, x):
    """단일 배치에 대한 예측"""
    pred, _ = self.forward(x)
    return pred

if __name__ == "__main__":
  print("Module Imported: Refactored CNN v1")
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  print("MNIST Data Loaded")

  # 데이터 전처리 및 cupy로 이동
  x_train = cp.asarray(x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0)
  x_test = cp.asarray(x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0)
  # One-hot 인코딩
  y_train_onehot = cp.eye(10)[y_train]
  y_test_onehot = cp.eye(10)[y_test]
  print("Data Preprocessing Completed")

  # CNN 모델 인스턴스 생성
  cnn = CNN(input_shape=(28, 28, 1), num_filters=[8, 16], filter_size=(3, 3), stride=1, pad=1,
            mlp_layer_sizes=[128, 10])
  print("Model Instantiated")

  # 학습 데이터 줄여서 테스트
  train_size = 5000
  x_train_sample = x_train[:train_size]
  y_train_sample = y_train_onehot[:train_size]

  print("Training started...")
  loss_history = cnn.train(x_train_sample, y_train_sample, lr=0.01, epochs=5, batch_size=64)
  print("Training Completed")

  # 정확도 평가
  print("Evaluating accuracy...")
  test_size = 1000
  predictions = cnn.predict(x_test[:test_size])
  predicted_labels = cp.argmax(predictions, axis=1)
  true_labels = y_test[:test_size]
  
  accuracy = cp.mean(predicted_labels == cp.asarray(true_labels))
  print(f"Accuracy on {test_size} test samples: {accuracy.get() * 100:.2f}%")

  # 손실 그래프
  plt.plot(loss_history)
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.title("Training Loss History")
  plt.grid(True)
  plt.show()

  # 예측 결과 샘플 확인
  fig, axes = plt.subplots(2, 5, figsize=(12, 6))
  for i, ax in enumerate(axes.flat):
    ax.imshow(x_test[i].get().reshape(28, 28), cmap='gray')
    pred_label = predicted_labels[i].get()
    true_label = true_labels[i]
    ax.set_title(f"Pred: {pred_label}\nTrue: {true_label}", color='green' if pred_label == true_label else 'red')
    ax.axis('off')
  plt.tight_layout()
  plt.show()
