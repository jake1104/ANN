
# Change Log
# 2025-10-09: 배치 처리, 임베딩 레이어

import cupy as cp
import numpy as np

class LSTM:
  def __init__(self, vocab_size, embedding_dim, n_h, n_y, lr=0.01):
    self.embedding_dim = embedding_dim
    self.n_h = n_h
    self.n_y = n_y

    # Embedding layer
    self.W_embed = cp.random.randn(vocab_size, embedding_dim) * 0.01

    # LSTM cell weights (input dimension is now embedding_dim)
    self.W_x = cp.random.randn(embedding_dim, 4 * n_h) * 0.01
    self.W_h = cp.random.randn(n_h, 4 * n_h) * 0.01
    self.b = cp.zeros((1, 4 * n_h))

    # Output layer weights
    self.Why = cp.random.randn(n_h, n_y) * 0.01
    self.by = cp.zeros((1, n_y))
    
    self.lr = lr

  def softmax(self, x):
    e = cp.exp(x - cp.max(x, axis=-1, keepdims=True))
    return e / cp.sum(e, axis=-1, keepdims=True)

  def sigmoid(self, x):
    return 1 / (1 + cp.exp(-x))

  def forward(self, X_batch_idx):
    # X_batch_idx shape: (batch_size, seq_len)
    batch_size, seq_len = X_batch_idx.shape
    
    # Embedding lookup
    X_embedded = self.W_embed[X_batch_idx] # (batch_size, seq_len, embedding_dim)

    hs = cp.zeros((batch_size, seq_len, self.n_h))
    ys = cp.zeros((batch_size, seq_len, self.n_y))
    caches = []
    
    h_prev = cp.zeros((batch_size, self.n_h))
    c_prev = cp.zeros((batch_size, self.n_h))

    for t in range(seq_len):
      x_t = X_embedded[:, t, :] # (batch_size, embedding_dim)
      
      gates = x_t @ self.W_x + h_prev @ self.W_h + self.b
      
      i_t = self.sigmoid(gates[:, :self.n_h])
      f_t = self.sigmoid(gates[:, self.n_h:2*self.n_h])
      o_t = self.sigmoid(gates[:, 2*self.n_h:3*self.n_h])
      g_t = cp.tanh(gates[:, 3*self.n_h:])
      
      c_t = f_t * c_prev + i_t * g_t
      h_t = o_t * cp.tanh(c_t)
      
      hs[:, t, :] = h_t
      ys[:, t, :] = self.softmax(h_t @ self.Why + self.by)
      
      caches.append((x_t, h_prev, c_prev, i_t, f_t, o_t, g_t, c_t, h_t))
      h_prev, c_prev = h_t, c_t
      
    # Store index for embedding backward pass
    caches.append(X_batch_idx)
    return hs, ys, caches

  def backward(self, Y_batch, ys, caches):
    batch_size, seq_len, _ = Y_batch.shape
    X_batch_idx = caches.pop()
    
    dW_embed = cp.zeros_like(self.W_embed)
    dW_x = cp.zeros_like(self.W_x)
    dW_h = cp.zeros_like(self.W_h)
    db = cp.zeros_like(self.b)
    dWhy = cp.zeros_like(self.Why)
    dby = cp.zeros_like(self.by)
    
    dh_next = cp.zeros((batch_size, self.n_h))
    dc_next = cp.zeros((batch_size, self.n_h))
    dX_embedded = cp.zeros((batch_size, seq_len, self.embedding_dim))

    dy = ys - Y_batch

    for t in reversed(range(seq_len)):
      dy_t = dy[:, t, :]
      x_t, h_prev, c_prev, i_t, f_t, o_t, g_t, c_t, h_t = caches[t]
      
      dWhy += h_t.T @ dy_t
      dby += cp.sum(dy_t, axis=0, keepdims=True)
      
      dh = dy_t @ self.Why.T + dh_next
      
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
      db += cp.sum(dgates, axis=0, keepdims=True)
      
      dh_next = dgates @ self.W_h.T
      dc_next = dc * f_t
      dX_embedded[:, t, :] = dgates @ self.W_x.T

    # Accumulate gradients for embedding layer
    cp.add.at(dW_embed, X_batch_idx, dX_embedded)

    # Clip gradients
    for dparam in [dW_embed, dW_x, dW_h, db, dWhy, dby]:
      cp.clip(dparam, -5, 5, out=dparam)

    # Update parameters
    self.W_embed -= self.lr * dW_embed / batch_size
    self.W_x -= self.lr * dW_x / batch_size
    self.W_h -= self.lr * dW_h / batch_size
    self.b -= self.lr * db / batch_size
    self.Why -= self.lr * dWhy / batch_size
    self.by -= self.lr * dby / batch_size

  def compute_loss(self, Y_batch, ys):
    batch_size = Y_batch.shape[0]
    loss = -cp.sum(Y_batch * cp.log(ys + 1e-9)) / batch_size
    return loss

  def train(self, batches, epochs=1000, print_interval=100):
    for epoch in range(1, epochs + 1):
      total_loss = cp.array(0.0)
      for X_batch_idx, Y_batch in batches:
        hs, ys, caches = self.forward(X_batch_idx)
        total_loss += self.compute_loss(Y_batch, ys)
        self.backward(Y_batch, ys, caches)
      
      if epoch % print_interval == 0:
        avg_loss = (total_loss / len(batches)).get()
        print(f"Epoch {epoch}/{epochs} | Avg Loss: {avg_loss:.4f}")

  def predict(self, seed_sequence, item_to_idx, idx_to_item, max_len, end_idx):
    h = cp.zeros((1, self.n_h))
    c = cp.zeros((1, self.n_h))
    
    current_idx = -1
    for item in seed_sequence:
      current_idx = item_to_idx[item]
      x_embedded = self.W_embed[current_idx].reshape(1, -1)
      
      gates = x_embedded @ self.W_x + h @ self.W_h + self.b
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

      result_sequence.append(idx_to_item[idx])
      current_idx = idx
      x_embedded = self.W_embed[current_idx].reshape(1, -1)
      
      gates = x_embedded @ self.W_x + h @ self.W_h + self.b
      i = self.sigmoid(gates[:, :self.n_h])
      f = self.sigmoid(gates[:, self.n_h:2*self.n_h])
      o = self.sigmoid(gates[:, 2*self.n_h:3*self.n_h])
      g = cp.tanh(gates[:, 3*self.n_h:])
      c = f * c + i * g
      h = o * cp.tanh(c)
      
    return result_sequence

# Helper function for batching with integer indices
def create_batches_for_embedding(sequences, item_to_idx, batch_size):
    vocab_size = len(item_to_idx)
    indexed_sequences = [[item_to_idx[item] for item in seq] for seq in sequences]
    indexed_sequences.sort(key=len, reverse=True)
    
    batches = []
    for i in range(0, len(indexed_sequences), batch_size):
        batch_seqs = indexed_sequences[i:i+batch_size]
        max_len = len(batch_seqs[0])
        
        X_batch_np = np.full((len(batch_seqs), max_len), 0, dtype=np.int32)
        Y_batch_np = np.full((len(batch_seqs), max_len, vocab_size), 0, dtype=np.float32)

        for j, seq in enumerate(batch_seqs):
            for k in range(len(seq) - 1):
                X_batch_np[j, k] = seq[k]
                Y_batch_np[j, k, seq[k+1]] = 1

        X_batch_idx = cp.array(X_batch_np)
        Y_batch = cp.array(Y_batch_np)
        batches.append((X_batch_idx, Y_batch))
        
    return batches

# ===============================
# 🧩 예제: 단어 완성
# ===============================
if __name__ == "__main__":
    end_token = "\n"
    sequences = ["Banana", "Apple", "Melon", "Grape", "Peach", "Orange", "Cherry", "Strawberry"]
    sequences = [s + end_token for s in sequences]

    corpus = "".join(sequences)
    vocab = sorted(list(set(corpus)))
    vocab_size = len(vocab)

    item_to_idx = {item: i for i, item in enumerate(vocab)}
    idx_to_item = {i: item for i, item in enumerate(vocab)}
    end_idx = item_to_idx[end_token]
    
    # Model and training parameters
    embedding_dim = 16 # New hyperparameter
    n_h = 64
    lr = 0.01
    epochs = 2000
    batch_size = 4
    print_interval = 200

    # Create batches with integer indices
    batches = create_batches_for_embedding(sequences, item_to_idx, batch_size)

    # Model initialization
    model = LSTM(vocab_size=vocab_size, embedding_dim=embedding_dim, n_h=n_h, n_y=vocab_size, lr=lr)
    
    # Training
    model.train(batches, epochs=epochs, print_interval=print_interval)

    print("\n🔮 단어 예측:")
    for seed in ["Ba", "Ap", "Me", "Gr", "Pe", "Or", "Ch", "St"]:
        predicted_sequence = model.predict(
            seed_sequence=seed, 
            item_to_idx=item_to_idx, 
            idx_to_item=idx_to_item, 
            max_len=10, 
            end_idx=end_idx
        )
        result_word = "".join(predicted_sequence)
        print(f"{seed} ➜ {result_word.strip()}")
