
# Change Log
# 2025-10-08: 미니배치 학습, 모델 저장/로드 안정성, adam 옵티마이저 추가

import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import os

def sigmoid(x):
  return 1 / (1 + cp.exp(-x))

def sigmoid_derivative(x):
  s = sigmoid(x)
  return s * (1 - s)

def softmax(x):
  # 배치 처리를 위해 axis=1 기준으로 max를 빼줌 (오버플로우 방지)
  e_x = cp.exp(x - cp.max(x, axis=1, keepdims=True))
  return e_x / cp.sum(e_x, axis=1, keepdims=True)

def cross_entropy_loss(pred, target):
  # log(0) 방지를 위해 작은 값(epsilon)을 더해줌
  pred = cp.clip(pred, 1e-9, 1 - 1e-9)
  # 배치 내의 모든 샘플에 대한 손실을 계산하고 평균을 냄
  return -cp.mean(cp.sum(target * cp.log(pred), axis=1))

class MLP:
  def __init__(self, layer_sizes=[2, 3, 2], activation_func=sigmoid, activation_func_derivative=sigmoid_derivative):
    self.layer_sizes = layer_sizes
    self.activation_func = activation_func
    self.activation_func_derivative = activation_func_derivative
    self.ws = []
    self.bs = []

    # Adam Optimizer parameters
    self.adam_params = {'t': 0, 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8}
    self.m, self.v = {}, {}

    # Xavier/Glorot 초기화: 가중치를 적절한 스케일로 초기화하여 학습 안정성 향상
    for i in range(len(layer_sizes) - 1):
      limit = cp.sqrt(6 / (layer_sizes[i] + layer_sizes[i + 1]))
      w = cp.random.uniform(-limit, limit, (layer_sizes[i], layer_sizes[i + 1]))
      b = cp.zeros((1, layer_sizes[i + 1]))
      self.ws.append(w)
      self.bs.append(b)

      # Initialize Adam moments
      self.m[f'W{i}'] = cp.zeros_like(w)
      self.v[f'W{i}'] = cp.zeros_like(w)
      self.m[f'b{i}'] = cp.zeros_like(b)
      self.v[f'b{i}'] = cp.zeros_like(b)

  def front_propagation(self, x):
    A = [x] # 활성화 값 저장 (입력층 포함)
    Z = []  # 활성화 함수를 통과하기 전의 값 저장
    # 은닉층
    for l in range(len(self.ws) - 1):
      z = cp.dot(A[-1], self.ws[l]) + self.bs[l]
      Z.append(z)
      a = self.activation_func(z)
      A.append(a)
    # 출력층
    z = cp.dot(A[-1], self.ws[-1]) + self.bs[-1]
    Z.append(z)
    a = softmax(z) # 출력층은 softmax 사용
    A.append(a)
    return A, Z

  def train_standalone(self, x, y, lr=0.1, epochs=10000, batch_size=32, print_interval=1000, target_loss=None):
    """
    미니배치 경사 하강법을 사용한 독립적인 모델 학습 함수.
    """
    loss_history = []
    for epoch in range(epochs):
      epoch_loss = 0
      num_batches = 0
      
      # 매 에포크마다 데이터 셔플링
      permutation = cp.random.permutation(x.shape[0])
      x_shuffled = x[permutation]
      y_shuffled = y[permutation]

      # 미니배치 생성 및 학습
      for i in range(0, x.shape[0], batch_size):
        x_batch = x_shuffled[i:i+batch_size]
        y_batch = y_shuffled[i:i+batch_size]
        
        # 1. 순전파
        A, Z = self.front_propagation(x_batch)
        pred = A[-1]
        
        # 2. 손실 계산
        loss = cross_entropy_loss(pred, y_batch)
        epoch_loss += loss.item()
        num_batches += 1

        # 3. 역전파 및 가중치 업데이트
        self.backward(pred - y_batch, A, Z, lr, x_batch.shape[0])

      avg_loss = epoch_loss / num_batches
      loss_history.append(avg_loss)

      if epoch % print_interval == 0:
        print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")

      if target_loss is not None and avg_loss <= target_loss:
        print(f"Early stopping at epoch {epoch}, Loss: {avg_loss:.6f}")
        break
        
    return loss_history

  def backward(self, dZ_final, A, Z, lr, batch_size):
    """
    단일 역전파 스텝을 수행하고 가중치를 업데이트합니다.
    CNN과 같은 상위 모델의 역전파 루프에 통합하기 위해 사용됩니다.
    
    :param dZ_final: 출력층에서 들어오는 초기 그라디언트 (pred - y)
    :param A: 순전파 시 계산된 활성화 값 리스트
    :param Z: 순전파 시 계산된 pre-활성화 값 리스트
    :param lr: 학습률
    :param batch_size: 현재 배치의 크기
    :return: 입력층에 대한 그라디언트 (상위 레이어로 전파하기 위함)
    """
    dW = [None] * len(self.ws)
    dB = [None] * len(self.ws)
    dZ = dZ_final

    # 출력층부터 입력층 방향으로 그라디언트 계산
    for l in reversed(range(len(self.ws))):
      dW[l] = cp.dot(A[l].T, dZ) / batch_size
      dB[l] = cp.mean(dZ, axis=0, keepdims=True)
      
      if l > 0:
        dA_prev = cp.dot(dZ, self.ws[l].T)
        dZ = dA_prev * self.activation_func_derivative(Z[l - 1])

    # CNN 레이어로 전달할 입력층에 대한 그라디언트
    d_input = cp.dot(dZ, self.ws[0].T)

    # Adam 옵티마이저로 가중치 및 편향 업데이트
    self.adam_params['t'] += 1
    t = self.adam_params['t']
    beta1, beta2, epsilon = self.adam_params['beta1'], self.adam_params['beta2'], self.adam_params['epsilon']

    for l in range(len(self.ws)):
      # 가중치 업데이트
      self.m[f'W{l}'] = beta1 * self.m[f'W{l}'] + (1 - beta1) * dW[l]
      self.v[f'W{l}'] = beta2 * self.v[f'W{l}'] + (1 - beta2) * (dW[l] ** 2)
      m_hat_w = self.m[f'W{l}'] / (1 - beta1 ** t)
      v_hat_w = self.v[f'W{l}'] / (1 - beta2 ** t)
      self.ws[l] -= lr * m_hat_w / (cp.sqrt(v_hat_w) + epsilon)

      # 편향 업데이트
      self.m[f'b{l}'] = beta1 * self.m[f'b{l}'] + (1 - beta1) * dB[l]
      self.v[f'b{l}'] = beta2 * self.v[f'b{l}'] + (1 - beta2) * (dB[l] ** 2)
      m_hat_b = self.m[f'b{l}'] / (1 - beta1 ** t)
      v_hat_b = self.v[f'b{l}'] / (1 - beta2 ** t)
      self.bs[l] -= lr * m_hat_b / (cp.sqrt(v_hat_b) + epsilon)
      
    return d_input

  def predict(self, x):
    A, _ = self.front_propagation(x)
    return A[-1]

  def save_model(self, filepath="mlp_model.npz"):
    """모델의 가중치, 편향, 구조(layer_sizes)를 npz 파일로 저장합니다."""
    params = {f'w{i}': w.get() for i, w in enumerate(self.ws)}
    params.update({f'b{i}': b.get() for i, b in enumerate(self.bs)})
    params['layer_sizes'] = np.array(self.layer_sizes)
    np.savez(filepath, **params)
    print(f"Model saved to {filepath}")

  def load_model(self, filepath="mlp_model.npz"):
    """npz 파일에서 모델을 로드합니다. 구조가 일치하는지 확인합니다."""
    if not os.path.exists(filepath):
      raise FileNotFoundError(f"Model file '{filepath}' not found.")
    
    data = np.load(filepath)
    
    # 모델 구조 확인
    if 'layer_sizes' in data:
      loaded_sizes = data['layer_sizes'].tolist()
      if loaded_sizes != self.layer_sizes:
        raise ValueError(f"Architecture mismatch: loaded model has sizes {loaded_sizes}, but this instance expects {self.layer_sizes}")
    else:
      print("Warning: 'layer_sizes' not found in model file. Assuming architecture is correct.")

    num_layers = len(self.layer_sizes) - 1
    if f'w{num_layers-1}' not in data:
        raise ValueError("Architecture mismatch: Number of layers in file does not match instance.")

    self.ws = [cp.array(data[f'w{i}']) for i in range(num_layers)]
    self.bs = [cp.array(data[f'b{i}']) for i in range(num_layers)]
    print(f"Model loaded from {filepath}")

  def get_parameters(self):
    """CNN 등 다른 모델과의 호환성을 위해 파라미터를 튜플 리스트로 반환"""
    return [(cp.asarray(w), cp.asarray(b)) for w, b in zip(self.ws, self.bs)]

  def set_parameters(self, params):
    """CNN 등 다른 모델로부터 파라미터를 설정"""
    if len(params) != len(self.ws):
      raise ValueError(f"Expected {len(self.ws)} parameter sets, got {len(params)}")
    self.ws = [cp.asarray(w) for w, b in params]
    self.bs = [cp.asarray(b) for w, b in params]
    print("MLP parameters loaded successfully.")
