import tkinter as tk
import numpy as np
import os
from .CNN_v04 import CNN
import cupy as cp
from scipy.ndimage import center_of_mass, shift

print("Module Imported")

# ========================================
# 모델 로드
# ========================================

# CNNj4.py에서 학습시킨 모델과 동일한 구조로 생성
cnn = CNN(
    input_shape=(28, 28, 1),
    num_filters=[32, 64],  # CNN_v04.py와 동일하게 설정
    filter_size=(3, 3),
    stride=1,
    pad=1,
    mlp_layer_sizes=[128, 10]
)

# 최종 모델 파일 경로
model_path = "mnist_model.npz"

if os.path.exists(model_path):
  print(f"Loading model from {model_path}...")
  cnn.load_model(model_path)
  print("✅ Model loaded successfully!")
else:
  raise FileNotFoundError(
      f"Model not found at {model_path}\n"
      f"Please train and save the model first by running CNNj4.py"
  )

# ========================================
# GUI: Pixel Editor
# ========================================
class PixelEditor:
  def __init__(self, master, model):
    self.model = model
    self.master = master
    self.master.title("Handwritten Digit Recognizer")

    self.canvas = tk.Canvas(self.master, width=280, height=280, bg='black')
    self.canvas.pack()

    self.pixel_size = 10
    self.pixels = np.zeros((28, 28), dtype=int)

    self.canvas.bind("<B1-Motion>", self.paint)
    self.canvas.bind("<ButtonRelease-1>", self.reset_title)

    self.clear_button = tk.Button(self.master, text="Clear", command=self.clear)
    self.clear_button.pack(side=tk.LEFT, padx=5, pady=5)

    self.predict_button = tk.Button(self.master, text="Predict", command=self.predict)
    self.predict_button.pack(side=tk.LEFT, padx=5, pady=5)

  def paint(self, event):
    x = event.x // self.pixel_size
    y = event.y // self.pixel_size
    # 3x3 브러시 효과
    for i in range(max(0, x - 1), min(28, x + 2)):
      for j in range(max(0, y - 1), min(28, y + 2)):
        dist_sq = (x - i)**2 + (y - j)**2
        # 중심은 255, 주변부는 거리에 따라 더 어둡게
        if dist_sq < 3:
          intensity = 255 - dist_sq * 60
          if self.pixels[j, i] < intensity:
            self.pixels[j, i] = int(intensity)
    self.update_canvas()

  def update_canvas(self):
    self.canvas.delete("all")
    for i in range(28):
      for j in range(28):
        color_val = self.pixels[j, i]
        color = f"#{color_val:02x}{color_val:02x}{color_val:02x}"
        self.canvas.create_rectangle(
            i * self.pixel_size, j * self.pixel_size,
            (i + 1) * self.pixel_size, (j + 1) * self.pixel_size,
            fill=color, outline=color
        )

  def clear(self):
    self.canvas.delete("all")
    self.pixels = np.zeros((28, 28), dtype=int)
    self.reset_title()

  def predict(self):
    pixels = self.pixels.astype(np.float32)

    # 입력이 비어 있는지 확인
    if np.sum(pixels) == 0:
        self.master.title("Canvas is empty!")
        return

    # 1. 질량 중심(center of mass)을 계산하여 중앙으로 이동
    cy, cx = center_of_mass(pixels)
    rows, cols = pixels.shape
    shiftx = cols/2.0 - cx
    shifty = rows/2.0 - cy
    
    # 이미지를 이동시켜 중앙에 정렬
    centered_pixels = shift(pixels, (shifty, shiftx))

    # 2. 모델 입력에 맞게 전처리
    input_image = centered_pixels / 255.0
    input_image_gpu = cp.asarray(input_image.reshape(1, 28, 28, 1))

    # 모델 예측
    output = self.model.predict(input_image_gpu)
    pred = cp.argmax(output, axis=1).get()[0]
    confidence = cp.max(output).get()

    print(f"✅ Predicted: {pred} | Confidence: {confidence:.4f}")
    self.master.title(f"Predicted: {pred} (Confidence: {confidence:.3f})")

  def reset_title(self, event=None):
    self.master.title("Handwritten Digit Recognizer")

# ========================================
# 실행
# ========================================
if __name__ == "__main__":
  root = tk.Tk()
  app = PixelEditor(root, cnn)
  root.mainloop()
