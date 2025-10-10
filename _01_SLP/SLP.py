
# Created at
# 2025-3 ~ 2025-4

import numpy as np

# 활성화 함수
binary_step_func = lambda x: 1 if x >= 0 else 0

class SLP:
  def __init__(self, n = 2):
    self.n : int = n
    self.ws : np.array = np.random.randn(n) / 100
    self.b : int = 0
  
  def predict(self, x):
    h = np.sum(np.dot(x, self.ws)) + self.b
    y = binary_step_func(h)
    return y
  
  def classic_train(self, x, y, lr, epochs):
    for _ in range(epochs):
      e = y - self.predict(x)
      self.ws += lr * np.dot(e, x)
      self.b += lr * e
  
  def train(self, x, y, lr, epochs):
    for _ in range(epochs):
      for i in range(len(x)):
        e = y[i] - self.predict(x[i])
        self.ws = self.ws + lr * e * x[i]
        self.b += lr * e

  def __str__(self):
    return f"SLP(n={self.n}, ws={self.ws}, b={self.b})"


# Example : X₁ | X₂
if __name__ == "__main__":
  import matplotlib.pyplot as plt
  from matplotlib.colors import ListedColormap
  
  x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
  y = np.array([0, 1, 1, 1])
  
  model = SLP(2)
  model.train(x, y, 0.1, 50)
  
  print(model.ws)
  print(model.b)
  
  # 분류 결과 확인 및 시각화
  cmap_background = ListedColormap(['#FFAAAA', '#AAAAFF'])
  cmap_points = ListedColormap(['#FF0000', '#0000FF'])
  
  # 결정 경계를 위한 그리드 생성
  x_min, x_max = -0.1, 1.1
  y_min, y_max = -0.1, 1.1
  xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
  
  Z = np.array([model.predict([i, j]) for i, j in zip(xx.ravel(), yy.ravel())])
  Z = Z.reshape(xx.shape)
  
  # x_line = np.linspace(0, 1, 100)
  # y_line = (-perceptron.b - x_line * perceptron.ws[0]) / perceptron.ws[1]
  # plt.plot(x_line, y_line, color='black', label='Decision Boundary')
  
  plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_background)
  plt.scatter(x[:, 0], x[:, 1], c=y, cmap=cmap_points, marker='o', s=100, label='Training Data')
  
  plt.xlabel('X₁')
  plt.ylabel('X₂')
  plt.legend()
  plt.show()
