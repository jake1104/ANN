
# Change Log
# 2025-8-30: save_model(), load_model() 추가
# 2025-8-31: get_parameters(), set_parameters() 추가

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
  e_x = cp.exp(x - cp.max(x, axis=1, keepdims=True))
  return e_x / cp.sum(e_x, axis=1, keepdims=True)

def cross_entropy_loss(pred, target):
  pred = cp.clip(pred, 1e-9, 1 - 1e-9)
  return -cp.mean(cp.sum(target * cp.log(pred), axis=1))

class MLP:
  def __init__(self, layer_sizes=[2, 3, 2], activation_func=sigmoid, activation_func_derivative=sigmoid_derivative):
    self.layer_sizes = layer_sizes
    self.activation_func = activation_func
    self.activation_func_derivative = activation_func_derivative
    self.ws = []
    self.bs = []

    for i in range(len(layer_sizes) - 1):
      limit = cp.sqrt(6 / (layer_sizes[i] + layer_sizes[i + 1]))
      w = cp.random.uniform(-limit, limit, (layer_sizes[i], layer_sizes[i + 1]))
      b = cp.zeros((1, layer_sizes[i + 1]))
      self.ws.append(w)
      self.bs.append(b)

  def front_propagation(self, x):
    A = [x]
    Z = []
    for l in range(len(self.ws) - 1):
      z = cp.dot(A[-1], self.ws[l]) + self.bs[l]
      Z.append(z)
      a = self.activation_func(z)
      A.append(a)
    z = cp.dot(A[-1], self.ws[-1]) + self.bs[-1]
    Z.append(z)
    a = softmax(z)
    A.append(a)
    return A, Z

  def back_propagation(self, x, y, lr=0.1, epochs=10000, print_interval=1000, target_loss=None):
    last_loss = None
    for epoch in range(epochs):
      A, Z = self.front_propagation(x)
      pred = A[-1]
      loss = cross_entropy_loss(pred, y)
      last_loss = loss

      if target_loss is not None and loss <= target_loss:
        print(f"Early stopping at epoch {epoch}, Loss: {loss:.6f}")
        break

      dz = pred - y
      dW = []
      dB = []

      dZ = dz
      for l in reversed(range(len(self.ws))):
        dw = cp.dot(A[l].T, dZ) / x.shape[0]
        db = cp.mean(dZ, axis=0, keepdims=True)
        dW.insert(0, dw)
        dB.insert(0, db)

        if l != 0:
          dA_prev = cp.dot(dZ, self.ws[l].T)
          dZ = dA_prev * self.activation_func_derivative(Z[l - 1])

      for l in range(len(self.ws)):
        self.ws[l] -= lr * dW[l]
        self.bs[l] -= lr * dB[l]

      if epoch % print_interval == 0:
        print(f"Epoch {epoch}, Loss: {loss:.6f}")
    return last_loss

  def predict(self, x):
    A, _ = self.front_propagation(x)
    return A[-1]

  def save_model(self, filepath="mlp_model.npz"):
    weights = [w.get() for w in self.ws]
    biases = [b.get() for b in self.bs]
    np.savez(filepath, **{f'w{i}': w for i, w in enumerate(weights)},
                        **{f'b{i}': b for i, b in enumerate(biases)})
    print(f"Model saved to {filepath}")

  def load_model(self, filepath="mlp_model.npz"):
    if not os.path.exists(filepath):
      raise FileNotFoundError(f"Model file '{filepath}' not found.")
    data = np.load(filepath)
    self.ws = [cp.array(data[f'w{i}']) for i in range(len(self.layer_sizes)-1)]
    self.bs = [cp.array(data[f'b{i}']) for i in range(len(self.layer_sizes)-1)]
    print(f"Model loaded from {filepath}")

  def get_parameters(self):
    """
    CNN.save_model()에서 호출: 모든 가중치와 편향을 튜플 리스트로 반환
    """
    return [(cp.asarray(w), cp.asarray(b)) for w, b in zip(self.ws, self.bs)]

  def set_parameters(self, params):
    """
    CNN.load_model()에서 호출: 저장된 파라미터를 다시 설정
    """
    if len(params) != len(self.ws):
      raise ValueError(f"Expected {len(self.ws)} parameter sets, got {len(params)}")
    self.ws = [cp.asarray(w) for w, b in params]
    self.bs = [cp.asarray(b) for w, b in params]
    print("MLP parameters loaded successfully.")


# ========================
# 테스트 코드 (XOR 문제)
# ========================
if __name__ == "__main__":
  x = cp.array([[0, 0],
                [0, 1],
                [1, 0],
                [1, 1]])
  y = cp.array([[1, 0],
                [0, 1],
                [0, 1],
                [1, 0]])

  model = MLP([2, 3, 2], sigmoid, sigmoid_derivative)
  model_path = "mlp_model.npz"

  if os.path.exists(model_path):
    model.load_model(model_path)
  else:
    print("Training model...")
    model.back_propagation(x, y, lr=0.5, epochs=10000, print_interval=1000)
    model.save_model(model_path)

  print("Final Predictions:\n", model.predict(x))

  # === 파라미터 추출 및 설정 테스트 ===
  print("\n--- Testing get/set_parameters ---")
  params = model.get_parameters()
  print(f"Got {len(params)} parameter sets.")

  # 새로운 모델 생성
  new_model = MLP([2, 3, 2], sigmoid, sigmoid_derivative)
  new_model.set_parameters(params)
  print("New model predictions after set_parameters():")
  print(new_model.predict(x))
  
  # 시각화
  import matplotlib.pyplot as plt

  # 입력 공간을 정의
  x_range = cp.linspace(-0.2, 1.2, 100)
  y_range = cp.linspace(-0.2, 1.2, 100)
  xx, yy = cp.meshgrid(x_range, y_range)
  zz = cp.zeros_like(xx)

  # 각 그리드 포인트에 대해 예측 (Class 1의 확률)
  for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
      input_point = cp.array([[xx[i, j], yy[i, j]]])  # (1, 2)
      pred = model.predict(input_point)               # softmax 출력: [P(0), P(1)]
      zz[i, j] = pred[0, 1]  # Class 1의 확률을 저장

  # Matplotlib으로 시각화 (cupy → numpy 변환 필요)
  plt.figure(figsize=(8, 6))
  plt.contourf(xx.get(), yy.get(), zz.get(), levels=50, cmap="coolwarm", alpha=0.7)
  plt.colorbar(label="Probability of Class 1")

  # 학습 데이터 포인트 시각화
  for i in range(len(x)):
    # 라벨 기준으로 색상 지정 (y는 one-hot 인코딩이므로 y[i, 1] 사용)
    color = 'red' if y[i, 1] == 1 else 'blue'  # Class 1: red, Class 0: blue
    plt.scatter(x[i, 0].get(), x[i, 1].get(), color=color, s=200, edgecolors='black', zorder=5)
    # 예측 확률 텍스트로 표시
    pred_val = model.predict(x[i:i+1])[0, 1].get()  # Class 1 확률
    plt.text(x[i, 0].get(), x[i, 1].get() + 0.05, f"{pred_val:.3f}", fontsize=12, ha='center', va='bottom', color='black')

  plt.xlabel("X1")
  plt.ylabel("X2")
  plt.title("XOR Problem - MLP Prediction (Class 1 Probability)")
  plt.grid(True, alpha=0.3)
  plt.xlim(-0.2, 1.2)
  plt.ylim(-0.2, 1.2)
  plt.tight_layout()
  plt.show()
