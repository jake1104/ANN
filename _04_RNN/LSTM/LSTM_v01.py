
# Created at
# 2025-10-08

import cupy as cp

class LSTM:
  def __init__(self, n_x, n_h, n_y, lr=0.05):
    self.n_h = n_h
    self.W_x = cp.random.randn(n_x, 4 * n_h) * 0.01
    self.W_h = cp.random.randn(n_h, 4 * n_h) * 0.01
    self.b = cp.zeros((1, 4 * n_h))
    self.Why = cp.random.randn(n_h, n_y) * 0.01
    self.by = cp.zeros((1, n_y))
    self.lr = lr

  def softmax(self, x):
    e = cp.exp(x - cp.max(x, axis=1, keepdims=True))
    return e / cp.sum(e, axis=1, keepdims=True)

  def sigmoid(self, x):
    return 1 / (1 + cp.exp(-x))

  def forward(self, X):
    hs, ys, caches = {}, {}, []
    h_prev = cp.zeros((1, self.n_h))
    c_prev = cp.zeros((1, self.n_h))

    for t in range(len(X)):
      x_t = X[t].reshape(1, -1)
      gates = x_t @ self.W_x + h_prev @ self.W_h + self.b
      
      i_t = self.sigmoid(gates[:, :self.n_h])
      f_t = self.sigmoid(gates[:, self.n_h:2*self.n_h])
      o_t = self.sigmoid(gates[:, 2*self.n_h:3*self.n_h])
      g_t = cp.tanh(gates[:, 3*self.n_h:])
      
      c_t = f_t * c_prev + i_t * g_t
      h_t = o_t * cp.tanh(c_t)
      y_t = self.softmax(h_t @ self.Why + self.by)
      
      hs[t] = h_t
      ys[t] = y_t
      caches.append((x_t, h_prev, c_prev, i_t, f_t, o_t, g_t, c_t, h_t))
      h_prev, c_prev = h_t, c_t
      
    return hs, ys, caches

  def backward(self, Y, ys, caches):
    dW_x = cp.zeros_like(self.W_x)
    dW_h = cp.zeros_like(self.W_h)
    db = cp.zeros_like(self.b)
    dWhy = cp.zeros_like(self.Why)
    dby = cp.zeros_like(self.by)
    
    dh_next = cp.zeros((1, self.n_h))
    dc_next = cp.zeros((1, self.n_h))

    for t in reversed(range(len(Y))):
      dy = ys[t] - Y[t].reshape(1, -1)
      x_t, h_prev, c_prev, i_t, f_t, o_t, g_t, c_t, h_t = caches[t]
      
      dWhy += h_t.T @ dy
      dby += dy
      
      dh = dy @ self.Why.T + dh_next
      
      do = dh * cp.tanh(c_t)
      dc = dc_next + dh * o_t * (1 - cp.tanh(c_t)**2)
      
      di = dc * g_t
      df = dc * c_prev
      dg = dc * i_t
      
      dgate_i = di * i_t * (1 - i_t)
      dgate_f = df * f_t * (1 - f_t)
      dgate_o = do * o_t * (1 - o_t)
      dgate_g = dg * (1 - g_t**2)
      
      dgates = cp.hstack((dgate_i, dgate_f, dgate_o, dgate_g))
      
      dW_x += x_t.T @ dgates
      dW_h += h_prev.T @ dgates
      db += dgates
      
      dh_next = dgates @ self.W_h.T
      dc_next = dc * f_t

    for dparam in [dW_x, dW_h, db, dWhy, dby]:
      cp.clip(dparam, -5, 5, out=dparam)

    self.W_x -= self.lr * dW_x
    self.W_h -= self.lr * dW_h
    self.b -= self.lr * db
    self.Why -= self.lr * dWhy
    self.by -= self.lr * dby

  def compute_loss(self, Y, ys):
    loss = 0
    for t in range(len(Y)):
      loss -= cp.sum(Y[t] * cp.log(ys[t] + 1e-8))
    return loss

  def prepare_sequences(self, sequences, item_to_idx):
    vocab_size = len(item_to_idx)
    X_train, Y_train = [], []
    for seq in sequences:
      X = [cp.eye(vocab_size)[item_to_idx[item]] for item in seq[:-1]]
      Y = [cp.eye(vocab_size)[item_to_idx[item]] for item in seq[1:]]
      X_train.append(X)
      Y_train.append(Y)
    return X_train, Y_train

  def train(self, X_train, Y_train, epochs=1000, print_interval=100):
    for epoch in range(1, epochs + 1):
      total_loss = cp.array(0.0)
      for X, Y in zip(X_train, Y_train):
        hs, ys, caches = self.forward(X)
        total_loss += self.compute_loss(Y, ys)
        self.backward(Y, ys, caches)
      if epoch % print_interval == 0:
        print(f"Epoch {epoch}/{epochs} | Loss: {total_loss.get():.4f}")

  def train_sequences(self, sequences, item_to_idx, epochs=1000, print_interval=100):
    X_train, Y_train = self.prepare_sequences(sequences, item_to_idx)
    self.train(X_train, Y_train, epochs, print_interval)

  def predict(self, seed_sequence, item_to_idx, idx_to_item, max_len, end_idx):
    vocab_size = len(item_to_idx)
    h = cp.zeros((1, self.n_h))
    c = cp.zeros((1, self.n_h))
    x = cp.zeros((1, vocab_size))

    for item in seed_sequence:
      x[:] = 0
      x[0, item_to_idx[item]] = 1
      gates = x @ self.W_x + h @ self.W_h + self.b
      i = self.sigmoid(gates[:, :self.n_h])
      f = self.sigmoid(gates[:, self.n_h:2*self.n_h])
      o = self.sigmoid(gates[:, 2*self.n_h:3*self.n_h])
      g = cp.tanh(gates[:, 3*self.n_h:])
      c = f * c + i * g
      h = o * cp.tanh(c)

    result_sequence = list(seed_sequence)
    for _ in range(max_len):
      y = self.softmax(h @ self.Why + self.by)
      idx = int(cp.argmax(y))

      if idx == end_idx:
        break

      item = idx_to_item[idx]
      result_sequence.append(item)
      
      x[:] = 0
      x[0, idx] = 1
      gates = x @ self.W_x + h @ self.W_h + self.b
      i = self.sigmoid(gates[:, :self.n_h])
      f = self.sigmoid(gates[:, self.n_h:2*self.n_h])
      o = self.sigmoid(gates[:, 2*self.n_h:3*self.n_h])
      g = cp.tanh(gates[:, 3*self.n_h:])
      c = f * c + i * g
      h = o * cp.tanh(c)
    return result_sequence


# ===============================
# 🧩 예제: 단어 완성 (Banana, Apple, Melon, Grape, Peach)
# ===============================
if __name__ == "__main__":
  # The module is now generic for sequences. This example uses characters.
  end_token = "\n"
  sequences = ["Banana", "Apple", "Melon", "Grape", "Peach"]
  sequences = [s + end_token for s in sequences]

  # Build vocabulary from data
  corpus = "".join(sequences)
  vocab = sorted(list(set(corpus)))
  vocab_size = len(vocab)

  item_to_idx = {item: i for i, item in enumerate(vocab)}
  idx_to_item = {i: item for item, i in item_to_idx.items()}
  end_idx = item_to_idx[end_token]

  # Model initialization
  model = LSTM(n_x=vocab_size, n_h=32, n_y=vocab_size, lr=0.05)
  
  # Training
  model.train_sequences(sequences, item_to_idx, epochs=2000, print_interval=200)

  print("\n🔮 단어 예측:")
  for seed in ["Ba", "Ap", "Me", "Gr", "Pe"]:
    predicted_sequence = model.predict(
        seed_sequence=seed, 
        item_to_idx=item_to_idx, 
        idx_to_item=idx_to_item, 
        max_len=10, 
        end_idx=end_idx
    )
    result_word = "".join(predicted_sequence)
    print(f"{seed} ➜ {result_word}")
