import argparse
import matplotlib.pyplot as plt
from .GRU_v01 import GRU, create_batches_for_embedding
import os
import numpy as np
import cupy as cp

usage_message = '''
================================================================================
GRU v1 (Adam Optimized) 사용법
================================================================================
- 스크립트를 그냥 실행하면, 기존에 학습된 'gru_model.npz'를 불러와 이어서 학습합니다.
- 처음부터 새로 학습하려면 `--force-train` 인자를 사용하세요.
- Adam 옵티마이저가 적용되어 학습이 더 안정적이고 빠릅니다.

- 사용 가능한 인자:
  --plot             : 학습 손실 그래프를 시각화합니다.
  --force-train      : 저장된 모델을 무시하고 처음부터 새로 학습을 시작합니다.
  --epochs <숫자>     : 학습할 에포크 수를 지정합니다. (기본값: 2000)
  --batch-size <숫자> : 배치 크기를 지정합니다. (기본값: 4)
  --lr <숫자>         : 학습률(learning rate)을 지정합니다. (기본값: 0.01)
  --target-loss <숫자>: 지정된 손실 값에 도달하면 학습을 조기 종료합니다.
  --load-path <경로>  : 지정된 경로에서 모델을 불러와 학습을 재개합니다.
  --save-path <경로>  : 학습 완료 후 모델을 지정된 경로에 저장합니다.

- 예시:
  # 모델을 불러와 500 에포크 추가 학습 후 결과 시각화
  python -m ANN._04_RNN.GRU.example --epochs 500 --plot

  # 특정 경로의 모델을 불러와 학습 후 다른 경로에 저장
  python -m ANN._04_RNN.GRU.example --load-path my_gru.npz --save-path new_gru.npz

  # 모델을 무시하고 처음부터 1000 에포크 재학습
  python -m ANN._04_RNN.GRU.example --force-train --epochs 1000
================================================================================
'''
print(usage_message)

parser = argparse.ArgumentParser(description="Train and evaluate the GRU v1 with Adam.")
parser.add_argument('--plot', action='store_true', help='Show matplotlib plot for loss.')
parser.add_argument('--force-train', action='store_true', help='Force retraining even if a model file exists.')
parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train. (Default: 2000)')
parser.add_argument('--batch-size', type=int, default=4, help='Batch size for training. (Default: 4)')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for Adam. (Default: 0.01)')
parser.add_argument('--target-loss', type=float, default=None, help='Target loss to stop training early.')
parser.add_argument('--load-path', type=str, default=None, help='Path to load model and resume training.')
parser.add_argument('--save-path', type=str, default=None, help='Path to save the final model.')
args = parser.parse_args()

end_token = "\n"
sequences = [s + end_token for s in ["Banana", "Apple", "Melon", "Grape", "Peach", "Orange", "Cherry", "Strawberry"]]
corpus = "".join(sequences)
vocab = sorted(list(set(corpus)))
vocab_size = len(vocab)
item_to_idx = {item: i for i, item in enumerate(vocab)}
idx_to_item = {i: item for item, i in item_to_idx.items()}
end_idx = item_to_idx[end_token]

embedding_dim=16
n_h=64

batches = create_batches_for_embedding(sequences, item_to_idx, args.batch_size)
model = GRU(vocab_size=vocab_size, embedding_dim=embedding_dim, n_h=n_h, n_y=vocab_size)

model_name = "gru_model"
final_model_path = args.save_path if args.save_path else f"{model_name}.npz"
load_model_path = args.load_path if args.load_path else final_model_path
checkpoint_dir = "checkpoints"
start_epoch = 0

if not args.force_train and os.path.exists(load_model_path):
    print(f"\nPrevious model found. Loading weights and training state from {load_model_path}")
    try:
        start_epoch = model.load_model(load_model_path)
    except (ValueError, KeyError, EOFError) as e:
        print(f"Error loading model: {e}. Starting from scratch.")
        start_epoch = 0
else:
    print("\nNo previous model found or --force-train specified. Training from scratch.")

print("\nStarting training...")
loss_history, last_epoch = model.train(batches, epochs=args.epochs, lr=args.lr, print_interval=100, 
                                       save_every=500, checkpoint_dir=checkpoint_dir, 
                                       model_name=model_name, start_epoch=start_epoch, target_loss=args.target_loss)

print(f"\nSaving final model to {final_model_path}...")
model.save_model(final_model_path, epoch=last_epoch)

if args.plot and loss_history:
    total_epochs_trained = range(start_epoch, start_epoch + len(loss_history))
    plt.plot(total_epochs_trained, loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss History (GRU v1 Adam)")
    plt.grid(True)
    plt.show()

print("\n🔮 Predicting with final model:")
for seed in ["Ba", "Ap", "Me", "Gr", "Pe", "Or", "Ch", "St"]:
    print(f"{seed} ➜ {model.predict(seed, item_to_idx, idx_to_item, 10, end_idx).strip()}")
