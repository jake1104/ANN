
# Change Log
# 2025-7-29: cupy로 변경

import cupy as cp
import math
import matplotlib.pyplot as plt

# 활성화 함수
linear_activation_func = lambda x: x
linear_activation_func_derivative = lambda x: 1

sigmoid = lambda x: 1 / (1 + cp.exp(-x))
sigmoid_derivative = lambda x: sigmoid(x) * (1 - sigmoid(x))

def softmax(x):
  exp_a = cp.exp(x - cp.max(x))  # 안정성 향상
  return exp_a / cp.sum(exp_a)

def softmax_derivative(x):
  s = softmax(x)
  return s * (1 - s)

hyperbolic_tangent = lambda x: (cp.exp(x) - cp.exp(-x)) / (cp.exp(x) + cp.exp(-x))
hyperbolic_tangent_derivative = lambda x: 1 - hyperbolic_tangent(x) ** 2

def mse(pred, y):
  return cp.mean((pred - y) ** 2)

def ce(pred, y):
  pred = cp.clip(pred, 1e-9, 1 - 1e-9)
  return -cp.mean(y * cp.log(pred) + (1 - y) * cp.log(1 - pred))

class MLP:
  def __init__(self, layer_sizes=[2, 3, 1], activation_func=sigmoid, activation_func_derivative=sigmoid_derivative, loss_function=mse):
    self.layer_sizes = layer_sizes
    self.activation_func = activation_func
    self.activation_func_derivative = activation_func_derivative
    self.loss_function = loss_function
    self.ws = []
    self.bs = []

    for i in range(len(layer_sizes) - 1):
      # 가중치 초기화 - Xavier 초기화로 변경 (성능향상 가능)
      limit = cp.sqrt(6 / (layer_sizes[i] + layer_sizes[i + 1]))
      w = cp.random.uniform(-limit, limit, (layer_sizes[i], layer_sizes[i + 1]))
      b = cp.zeros((1, layer_sizes[i + 1]))
      self.ws.append(w)
      self.bs.append(b)

  def front_propagation(self, x):
      A = x
      for l in range(len(self.ws)):
          A = self.activation_func(cp.dot(A, self.ws[l]) + self.bs[l])
      return A

  def back_propagation(self, x, y, lr=0.1, epochs=10000):
    for epoch in range(epochs):
      # 순전파: 전체 배치 한번에
      A = [x]
      Z = []
      for l in range(len(self.ws)):
        z = cp.dot(A[-1], self.ws[l]) + self.bs[l]
        a = self.activation_func(z)
        Z.append(z)
        A.append(a)

      # 손실 계산
      loss = self.loss_function(A[-1], y)

      # 출력층 오차
      dz = 2 * (A[-1] - y) * self.activation_func_derivative(Z[-1])

      dW = []
      dB = []
      dZ = dz

      # 역전파 (벡터화)
      for l in reversed(range(len(self.ws))):
        dw = cp.dot(A[l].T, dZ) / x.shape[0]  # 배치 평균
        db = cp.mean(dZ, axis=0, keepdims=True)  # 배치 평균
        dW.insert(0, dw)
        dB.insert(0, db)

        if l != 0:
          dA_prev = cp.dot(dZ, self.ws[l].T)
          dZ = dA_prev * self.activation_func_derivative(Z[l - 1])

      # 파라미터 업데이트
      for l in range(len(self.ws)):
        self.ws[l] -= lr * dW[l]
        self.bs[l] -= lr * dB[l]

      if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

if __name__ == "__main__":
  x = cp.array([[0, 0],
                [0, 1],
                [1, 0],
                [1, 1]])
  y = cp.array([[0], [1], [1], [0]])

  model = MLP([2, 3, 1], sigmoid, sigmoid_derivative, mse)
  print("초기 예측:\n", model.front_propagation(x))
  model.back_propagation(x, y, lr=0.5, epochs=10000)
  print("학습 후 예측:\n", model.front_propagation(x))

  # 시각화
  import matplotlib.pyplot as plt
  x_range = cp.linspace(-0.2, 1.2, 100)
  y_range = cp.linspace(-0.2, 1.2, 100)
  xx, yy = cp.meshgrid(x_range, y_range)
  zz = cp.zeros_like(xx)

  for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
      icput_point = cp.array([[xx[i, j], yy[i, j]]])
      zz[i, j] = model.front_propagation(icput_point)[0, 0]

  plt.contourf(xx.get(), yy.get(), zz.get(), levels=50, cmap="coolwarm", alpha=0.7)

  for i in range(len(x)):
    color = 'red' if y[i] == 1 else 'blue'
    plt.scatter(x[i, 0].get(), x[i, 1].get(), color=color, s=200, edgecolors='black')
    pred_val = model.front_propagation(x[i].reshape(1, -1))[0, 0]
    plt.text(x[i, 0].get(), x[i, 1].get() + 0.05, f"{pred_val:.3f}", fontsize=12, ha='center')

  plt.xlabel("X1")
  plt.ylabel("X2")
  plt.title("XOR Problem - MLP Prediction")
  plt.grid(True)
  plt.show()
