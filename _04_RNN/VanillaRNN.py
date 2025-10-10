
# RNN (Recurrent Neural Network, 순환 신경망)
# 2025-10-08: Vanilla RNN 구현

import cupy as cp

class VanillaRNN:
  def __init__(self, n_x, n_h, n_y, lr=0.05):
    self.Wxh = cp.random.randn(n_x, n_h) * 0.01
    self.Whh = cp.random.randn(n_h, n_h) * 0.01
    self.Why = cp.random.randn(n_h, n_y) * 0.01
    self.bh = cp.zeros((1, n_h))
    self.by = cp.zeros((1, n_y))
    self.lr = lr

  def softmax(self, x):
    e = cp.exp(x - cp.max(x, axis=1, keepdims=True))
    return e / cp.sum(e, axis=1, keepdims=True)

  def forward(self, X):
    hs, ys = {}, {}
    h_prev = cp.zeros((1, self.Whh.shape[0]))
    for t in range(len(X)):
      x_t = X[t].reshape(1, -1)
      h_t = cp.tanh(x_t @ self.Wxh + h_prev @ self.Whh + self.bh)
      y_t = self.softmax(h_t @ self.Why + self.by)
      hs[t] = h_t
      ys[t] = y_t
      h_prev = h_t
    return hs, ys

  def backward(self, X, Y, hs, ys):
    dWxh = cp.zeros_like(self.Wxh)
    dWhh = cp.zeros_like(self.Whh)
    dWhy = cp.zeros_like(self.Why)
    dbh = cp.zeros_like(self.bh)
    dby = cp.zeros_like(self.by)
    dh_next = cp.zeros_like(hs[0])

    for t in reversed(range(len(X))):
      dy = ys[t] - Y[t].reshape(1, -1)
      dWhy += hs[t].T @ dy
      dby += dy
      dh = dy @ self.Why.T + dh_next
      dh_raw = (1 - hs[t] ** 2) * dh
      dbh += dh_raw
      dWxh += X[t].reshape(1, -1).T @ dh_raw
      if t > 0:
        dWhh += hs[t - 1].T @ dh_raw
      dh_next = dh_raw @ self.Whh.T

    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
      cp.clip(dparam, -5, 5, out=dparam)

    self.Wxh -= self.lr * dWxh
    self.Whh -= self.lr * dWhh
    self.Why -= self.lr * dWhy
    self.bh -= self.lr * dbh
    self.by -= self.lr * dby

  def compute_loss(self, Y, ys):
    loss = 0
    for t in range(len(Y)):
      loss -= cp.sum(Y[t] * cp.log(ys[t] + 1e-8))
    return loss

  def prepare_sequences(self, words, char_to_idx):
    vocab_size = len(char_to_idx)
    X_train, Y_train = [], []
    for word in words:
      X = [cp.eye(vocab_size)[char_to_idx[ch]] for ch in word[:-1]]
      Y = [cp.eye(vocab_size)[char_to_idx[ch]] for ch in word[1:]]
      X_train.append(X)
      Y_train.append(Y)
    return X_train, Y_train

  def train(self, X_train, Y_train, epochs=1000, print_interval=100):
    for epoch in range(1, epochs + 1):
      total_loss = 0
      for X, Y in zip(X_train, Y_train):
        hs, ys = self.forward(X)
        total_loss += self.compute_loss(Y, ys)
        self.backward(X, Y, hs, ys)
      if epoch % print_interval == 0:
        print(f"Epoch {epoch}/{epochs} | Loss: {total_loss:.4f}")

  def train_words(self, words, char_to_idx, epochs=1000, print_interval=100):
    X_train, Y_train = self.prepare_sequences(words, char_to_idx)
    self.train(X_train, Y_train, epochs, print_interval)

  def predict(self, seed_text, char_to_idx, idx_to_char, length):
    vocab_size = len(char_to_idx)
    h = cp.zeros((1, self.Whh.shape[0]))
    x = cp.zeros((1, vocab_size))
    for ch in seed_text:
      x[:] = 0
      x[0, char_to_idx[ch]] = 1
      h = cp.tanh(x @ self.Wxh + h @ self.Whh + self.bh)
    result = seed_text
    for _ in range(length):
      y = self.softmax(h @ self.Why + self.by)
      idx = int(cp.argmax(y))  # <--- 오류 수정: 정수로 변환
      ch = idx_to_char[idx]
      result += ch
      x[:] = 0
      x[0, idx] = 1
      h = cp.tanh(x @ self.Wxh + h @ self.Whh + self.bh)
    return result


# ===============================
# 🧩 예제: 단어 완성 (Banana, Apple, Melon, Grape, Peach)
# ===============================
if __name__ == "__main__":
  words = ["Banana", "Apple", "Melon", "Grape", "Peach"]
  text = "".join(words)
  chars = sorted(list(set(text)))
  vocab_size = len(chars)

  char_to_idx = {ch: i for i, ch in enumerate(chars)}
  idx_to_char = {i: ch for ch, i in char_to_idx.items()}

  rnn = VanillaRNN(n_x=vocab_size, n_h=32, n_y=vocab_size, lr=0.05)
  rnn.train_words(words, char_to_idx, epochs=2000, print_interval=200)

  print("\n🔮 단어 예측:")
  for seed in ["Ba", "Ap", "Me", "Gr", "Pe"]:
    result = rnn.predict(seed, char_to_idx, idx_to_char, length=4)
    print(f"{seed} ➜ {result}")
