import argparse
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import cupy as cp
import os
from .CNN_v04 import CNN

usage_message = '''
================================================================================
Refactored CNN v4.4 (Adam Optimized) 사용법
================================================================================
- 스크립트를 그냥 실행하면, 기존에 학습된 'mnist_model.npz'를 불러와 이어서 학습합니다.
- 처음부터 새로 학습하려면 `--force-train` 인자를 사용하세요.
- Adam 옵티마이저가 적용되어 학습이 더 안정적이고 빠릅니다.

- 사용 가능한 인자:
  --plot             : 학습 손실 그래프와 예측 결과 이미지를 시각화합니다.
  --force-train      : 저장된 모델을 무시하고 처음부터 새로 학습을 시작합니다.
  --epochs <숫자>     : 학습할 에포크 수를 지정합니다. (기본값: 10)
  --batch-size <숫자> : 배치 크기를 지정합니다. (기본값: 64)
  --lr <숫자>         : 학습률(learning rate)을 지정합니다. (기본값: 0.001)
  --target-loss <숫자>: 지정된 손실 값에 도달하면 학습을 조기 종료합니다.
  --load-path <경로>  : 지정된 경로에서 모델을 불러와 학습을 재개합니다.
  --save-path <경로>  : 학습 완료 후 모델을 지정된 경로에 저장합니다.

- 예시:
  # 모델을 불러와 5 에포크 추가 학습 후 결과 시각화
  python CNN_v04.py --epochs 5 --plot

  # 특정 경로의 모델을 불러와 학습 후 다른 경로에 저장
  python CNN_v04.py --load-path my_model.npz --save-path new_model.npz

  # 모델을 무시하고 처음부터 20 에포크 재학습
  python CNN_v04.py --force-train --epochs 20
================================================================================
'''
print(usage_message)

parser = argparse.ArgumentParser(description="Train and evaluate the Refactored CNN v4 with Adam.")
parser.add_argument('--plot', action='store_true', help='Show matplotlib plots for loss and predictions.')
parser.add_argument('--force-train', action='store_true', help='Force retraining even if a model file exists.')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train. (Default: 10)')
parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training. (Default: 64)')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for Adam. (Default: 0.001)')
parser.add_argument('--target-loss', type=float, default=None, help='Target loss to stop training early.')
parser.add_argument('--load-path', type=str, default=None, help='Path to load model and resume training.')
parser.add_argument('--save-path', type=str, default=None, help='Path to save the final model.')
args = parser.parse_args()

print("Module Imported: Refactored CNN v4.4 (Adam Optimized)")
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("MNIST Data Loaded")

x_train = cp.asarray(x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0)
x_test = cp.asarray(x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0)
y_train_onehot = cp.eye(10)[y_train]
print("Data Preprocessing Completed")

# 모델 용량 증가: num_filters=[32, 64]
cnn = CNN(input_shape=(28, 28, 1), num_filters=[32, 64], filter_size=(3, 3), stride=1, pad=1,
          mlp_layer_sizes=[128, 10])
print("Model Instantiated with increased capacity: [32, 64] filters.")

model_name = "mnist_model"
final_model_path = args.save_path if args.save_path else f"{model_name}.npz"
load_model_path = args.load_path if args.load_path else final_model_path
start_epoch = 0

if not args.force_train and os.path.exists(load_model_path):
  print(f"\nPrevious model found. Loading weights and training state from {load_model_path}")
  try:
      start_epoch = cnn.load_model(load_model_path)
  except (ValueError, KeyError) as e:
      print(f"Error loading model: {e}. Starting from scratch.")
      start_epoch = 0
else:
  print("\nNo previous model found or --force-train specified. Training from scratch.")

print("\nStarting training...")
# 전체 학습 데이터 사용
train_size = 60000
x_train_sample = x_train[:train_size]
y_train_sample = y_train_onehot[:train_size]
loss_history, last_epoch = cnn.train(x_train_sample, y_train_sample, lr=args.lr, epochs=args.epochs, 
                                     batch_size=args.batch_size, save_every=1, target_loss=args.target_loss, 
                                     model_name=model_name, start_epoch=start_epoch)

print(f"\nSaving final model to {final_model_path}...")
cnn.save_model(final_model_path, epoch=last_epoch)

if args.plot and loss_history:
  total_epochs_trained = range(start_epoch, start_epoch + len(loss_history))
  plt.plot(total_epochs_trained, loss_history)
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.title("Training Loss History (CNN v4.4 Adam)")
  plt.grid(True)
  plt.show()

print("\nEvaluating accuracy...")
test_size = 1000
predictions = cnn.predict(x_test[:test_size])
predicted_labels = cp.argmax(predictions, axis=1)
true_labels = y_test[:test_size]
accuracy = cp.mean(predicted_labels == cp.asarray(true_labels))
print(f"Accuracy on {test_size} test samples: {accuracy.get() * 100:.2f}%")

if args.plot:
  print("Displaying prediction samples...")
  fig, axes = plt.subplots(2, 5, figsize=(12, 6))
  for i, ax in enumerate(axes.flat):
      ax.imshow(x_test[i].get().reshape(28, 28), cmap='gray')
      pred_label = predicted_labels[i].get()
      true_label = true_labels[i]
      ax.set_title(f"Pred: {pred_label}\nTrue: {true_label}", color='green' if pred_label == true_label else 'red')
      ax.axis('off')
  plt.tight_layout()
  plt.show()
