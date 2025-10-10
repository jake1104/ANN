import argparse
import cupy as cp
import matplotlib.pyplot as plt
import os
from .MLP_v04 import MLP, sigmoid, sigmoid_derivative

parser = argparse.ArgumentParser(description="Train and evaluate the MLP v4 for the XOR problem.")
parser.add_argument('--load-path', type=str, default=None, help='Path to load a pre-trained model.')
parser.add_argument('--save-path', type=str, default="mlp_model_v4.npz", help='Path to save the trained model.')
args = parser.parse_args()

x = cp.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = cp.array([[1, 0], [0, 1], [0, 1], [1, 0]]) # One-hot encoding for 2 classes

model = MLP([2, 4, 2], sigmoid, sigmoid_derivative) # 4개의 뉴런을 가진 은닉층

load_path = args.load_path if args.load_path is not None else args.save_path

should_train = True
if os.path.exists(load_path):
  try:
      print(f"Loading pre-trained model from {load_path}...")
      model.load_model(load_path)
      should_train = False
  except (ValueError, FileNotFoundError, KeyError) as e: # Added KeyError for robustness
      print(f"Error loading model: {e}. Training new model...")
      should_train = True

if should_train:
  print("Training new model...")
  # 미니배치 학습 (XOR은 데이터가 작아 batch_size=4가 풀배치와 동일)
  model.train_standalone(x, y, lr=0.01, epochs=10000, batch_size=4, print_interval=1000) # Adam은 lr을 더 작게
  model.save_model(args.save_path)

# 최종 예측 결과 확인
final_predictions = model.predict(x)
print("\nFinal Predictions:\n", final_predictions)

# === 파라미터 추출 및 설정 테스트 ===
print("\n--- Testing get/set_parameters ---")
params = model.get_parameters()
print(f"Got {len(params)} parameter sets.")
new_model = MLP([2, 4, 2])
new_model.set_parameters(params)
print("New model predictions after set_parameters():")
print(new_model.predict(x))

# === 결정 경계 시각화 (벡터화된 효율적인 코드) ===
print("\nGenerating decision boundary plot...")
plt.figure(figsize=(8, 6))

# 1. 시각화할 그리드 생성
x_range = cp.linspace(-0.2, 1.2, 100)
y_range = cp.linspace(-0.2, 1.2, 100)
xx, yy = cp.meshgrid(x_range, y_range)

# 2. 그리드 포인트를 하나의 배치로 만듦
grid_points = cp.c_[xx.ravel(), yy.ravel()]

# 3. 전체 배치에 대해 예측 한 번만 수행
grid_predictions = model.predict(grid_points)

# 4. Class 1에 대한 확률을 다시 그리드 형태로 변환
zz = grid_predictions[:, 1].reshape(xx.shape)

# 5. 등고선 플롯으로 결정 경계 시각화
plt.contourf(xx.get(), yy.get(), zz.get(), levels=50, cmap="coolwarm", alpha=0.7)
plt.colorbar(label="Probability of Class 1")

# 6. 학습 데이터 포인트 및 예측 확률 시각화
class_0_points = x[y[:, 0] == 1]
class_1_points = x[y[:, 1] == 1]
plt.scatter(class_0_points[:, 0].get(), class_0_points[:, 1].get(), color='blue', s=200, edgecolors='black', zorder=5, label='Class 0')
plt.scatter(class_1_points[:, 0].get(), class_1_points[:, 1].get(), color='red', s=200, edgecolors='black', zorder=5, label='Class 1')

# 각 점 위에 예측 확률(Class 1) 텍스트 표시
for i in range(x.shape[0]):
  point_x = x[i, 0].get()
  point_y = x[i, 1].get()
  prob_class_1 = final_predictions[i, 1].get()
  plt.text(point_x, point_y + 0.05, f"{prob_class_1:.3f}", fontsize=12, ha='center', va='bottom', color='black')

plt.xlabel("X1")
plt.ylabel("X2")
plt.title("XOR Problem - MLP Decision Boundary (v4)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(-0.2, 1.2)
plt.ylim(-0.2, 1.2)
plt.tight_layout()
plt.show()
print("Plot displayed.")
