# Change Log
# 2025-10-09: Adam 옵티마이저, 체크포인트, 학습률 스케줄러

import cupy as cp
import numpy as np
import os

class LSTM:
  @staticmethod
  @cp.fuse()
  def _fused_forward_cell(g_i, g_f, g_o, g_g, c_prev):
    i_t = 1 / (1 + cp.exp(-g_i)); f_t = 1 / (1 + cp.exp(-g_f)); o_t = 1 / (1 + cp.exp(-g_o)); g_t = cp.tanh(g_g)
    c_t = f_t * c_prev + i_t * g_t
    h_t = o_t * cp.tanh(c_t)
    return h_t, c_t, i_t, f_t, o_t, g_t

  @staticmethod
  @cp.fuse()
  def _fused_backward_cell(dh, dc_next, c_prev, c_t, i_t, f_t, o_t, g_t):
    tanh_c_t = cp.tanh(c_t); do = dh * tanh_c_t; dc = dc_next + dh * o_t * (1 - tanh_c_t**2)
    di = dc * g_t; df = dc * c_prev; dg = dc * i_t
    dgate_i = di * i_t * (1 - i_t); dgate_f = df * f_t * (1 - f_t)
    dgate_o = do * o_t * (1 - o_t); dgate_g = dg * (1 - g_t**2)
    dc_next_new = dc * f_t
    return dgate_i, dgate_f, dgate_o, dgate_g, dc_next_new

  def __init__(self, vocab_size, embedding_dim, n_h, n_y):
    # Xavier Init
    limit_embed = np.sqrt(6.0 / (vocab_size + embedding_dim)); limit_wx = np.sqrt(6.0 / (embedding_dim + 4 * n_h))
    limit_wh = np.sqrt(6.0 / (n_h + 4 * n_h)); limit_why = np.sqrt(6.0 / (n_h + n_y))

    # Model Parameters
    self.params = {
        'W_embed': cp.random.uniform(-limit_embed, limit_embed, (vocab_size, embedding_dim), dtype=cp.float32),
        'W_x': cp.random.uniform(-limit_wx, limit_wx, (embedding_dim, 4 * n_h), dtype=cp.float32),
        'W_h': cp.random.uniform(-limit_wh, limit_wh, (n_h, 4 * n_h), dtype=cp.float32),
        'b': cp.zeros((1, 4 * n_h), dtype=cp.float32),
        'Why': cp.random.uniform(-limit_why, limit_why, (n_h, n_y), dtype=cp.float32),
        'by': cp.zeros((1, n_y), dtype=cp.float32)
    }

    # Adam Optimizer Parameters
    self.m = {p: cp.zeros_like(self.params[p]) for p in self.params}
    self.v = {p: cp.zeros_like(self.params[p]) for p in self.params}
    self.t = 0
    self.beta1 = 0.9
    self.beta2 = 0.999
    self.epsilon = 1e-8

  def softmax(self, x): return cp.exp(x - cp.max(x, axis=-1, keepdims=True)) / cp.sum(cp.exp(x - cp.max(x, axis=-1, keepdims=True)), axis=-1, keepdims=True)
  def sigmoid(self, x): return 1 / (1 + cp.exp(-x))

  def forward(self, X_batch_idx):
    batch_size, seq_len = X_batch_idx.shape
    X_embedded = self.params['W_embed'][X_batch_idx]
    hs = cp.zeros((batch_size, seq_len, self.params['W_h'].shape[0]), dtype=cp.float32)
    ys = cp.zeros((batch_size, seq_len, self.params['Why'].shape[1]), dtype=cp.float32)
    caches = []
    h_prev = cp.zeros((batch_size, self.params['W_h'].shape[0]), dtype=cp.float32)
    c_prev = cp.zeros((batch_size, self.params['W_h'].shape[0]), dtype=cp.float32)

    for t in range(seq_len):
      x_t = X_embedded[:, t, :]
      gates = x_t @ self.params['W_x'] + h_prev @ self.params['W_h'] + self.params['b']
      h_t, c_t, i_t, f_t, o_t, g_t = self._fused_forward_cell(gates[:, :self.params['W_h'].shape[0]], gates[:, self.params['W_h'].shape[0]:2*self.params['W_h'].shape[0]], gates[:, 2*self.params['W_h'].shape[0]:3*self.params['W_h'].shape[0]], gates[:, 3*self.params['W_h'].shape[0]:], c_prev)
      ys[:, t, :] = self.softmax(h_t @ self.params['Why'] + self.params['by'])
      caches.append((x_t, h_prev, c_prev, i_t, f_t, o_t, g_t, c_t, h_t)); h_prev, c_prev = h_t, c_t
    caches.append(X_batch_idx)
    return hs, ys, caches

  def backward(self, Y_batch, ys, caches):
    batch_size, seq_len, _ = Y_batch.shape; X_batch_idx = caches.pop()
    n_h, n_y, embedding_dim = self.params['W_h'].shape[0], self.params['Why'].shape[1], self.params['W_embed'].shape[1]
    grads = {p: cp.zeros_like(self.params[p]) for p in self.params}
    dh_next = cp.zeros((batch_size, n_h), dtype=cp.float32); dc_next = cp.zeros((batch_size, n_h), dtype=cp.float32)
    dX_embedded = cp.zeros((batch_size, seq_len, embedding_dim), dtype=cp.float32)
    dy = ys - Y_batch
    for t in reversed(range(seq_len)):
      dy_t = dy[:, t, :]; x_t, h_prev, c_prev, i_t, f_t, o_t, g_t, c_t, h_t = caches[t]
      grads['Why'] += h_t.T @ dy_t; grads['by'] += cp.sum(dy_t, axis=0, keepdims=True)
      dh = dy_t @ self.params['Why'].T + dh_next
      dgate_i, dgate_f, dgate_o, dgate_g, dc_next = self._fused_backward_cell(dh, dc_next, c_prev, c_t, i_t, f_t, o_t, g_t)
      dgates = cp.hstack((dgate_i, dgate_f, dgate_o, dgate_g))
      grads['W_x'] += x_t.T @ dgates; grads['W_h'] += h_prev.T @ dgates; grads['b'] += cp.sum(dgates, axis=0, keepdims=True)
      dh_next = dgates @ self.params['W_h'].T; dX_embedded[:, t, :] = dgates @ self.params['W_x'].T
    cp.add.at(grads['W_embed'], X_batch_idx, dX_embedded)
    for p in self.params: cp.clip(grads[p], -5, 5, out=grads[p])
    return grads

  def update_parameters_adam(self, grads, lr):
    self.t += 1
    for p in self.params:
        self.m[p] = self.beta1 * self.m[p] + (1 - self.beta1) * grads[p]
        self.v[p] = self.beta2 * self.v[p] + (1 - self.beta2) * (grads[p]**2)
        m_hat = self.m[p] / (1 - self.beta1**self.t)
        v_hat = self.v[p] / (1 - self.beta2**self.t)
        self.params[p] -= lr * m_hat / (cp.sqrt(v_hat) + self.epsilon)

  def compute_loss(self, Y_batch, ys): return -cp.sum(Y_batch * cp.log(ys + 1e-9)) / Y_batch.shape[0]

  def train(self, batches, epochs, lr, print_interval=100, save_every=500, checkpoint_dir="checkpoints", model_name="lstm_model", start_epoch=0, target_loss=None):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        
    loss_history = []
    current_lr = lr

    for epoch in range(start_epoch, start_epoch + epochs):
      # Learning Rate Scheduling
      if epoch > start_epoch and (epoch - start_epoch) % 500 == 0:
          current_lr /= 2.0
          print(f"\nLearning rate decayed to {current_lr}\n")

      total_loss = cp.array(0.0, dtype=cp.float32)
      num_batches = 0
      for X_batch_idx, Y_batch in batches:
        hs, ys, caches = self.forward(X_batch_idx)
        loss = self.compute_loss(Y_batch, ys)
        total_loss += loss
        grads = self.backward(Y_batch, ys, caches)
        self.update_parameters_adam(grads, current_lr)
        num_batches += 1
        
      avg_loss = (total_loss / num_batches).get()
      loss_history.append(avg_loss)

      if (epoch + 1) % print_interval == 0:
        print(f"Epoch {epoch + 1}/{start_epoch + epochs} | LR: {current_lr:.6f} | Avg Loss: {avg_loss:.4f}")

      if (epoch + 1) % save_every == 0:
        self.save_model(os.path.join(checkpoint_dir, f"{model_name}_epoch_{epoch+1}.npz"), epoch=epoch + 1)

      if target_loss is not None and avg_loss <= target_loss:
        print(f"Target loss {target_loss} reached. Early stopping at epoch {epoch + 1}.")
        break
        
    return loss_history, start_epoch + len(loss_history)

  def save_model(self, path="model.npz", epoch=0):
    data_to_save = {p: self.params[p].get() for p in self.params}
    data_to_save.update({f'm_{p}': self.m[p].get() for p in self.m})
    data_to_save.update({f'v_{p}': self.v[p].get() for p in self.v})
    data_to_save['t'] = self.t
    data_to_save['epoch'] = epoch
    
    np.savez(path, **data_to_save)
    print(f"Model saved to {path}")

  def load_model(self, path):
    if not os.path.exists(path):
        print(f"Model file not found: {path}. Starting from scratch.")
        return 0
    
    data = np.load(path, allow_pickle=True)
    
    for p in self.params:
        if p in data:
            self.params[p] = cp.asarray(data[p])
    
    for p in self.m:
        m_key = f'm_{p}'
        if m_key in data:
            self.m[p] = cp.asarray(data[m_key])

    for p in self.v:
        v_key = f'v_{p}'
        if v_key in data:
            self.v[p] = cp.asarray(data[v_key])

    self.t = int(data['t'].item()) if 't' in data else 0
    start_epoch = int(data['epoch'].item()) if 'epoch' in data else 0
    
    print(f"Model loaded from {path}. Resuming from epoch {start_epoch + 1}")
    return start_epoch

  def predict(self, seed, item_to_idx, idx_to_item, max_len, end_idx):
    n_h = self.params['W_h'].shape[0]; h = cp.zeros((1, n_h), dtype=cp.float32); c = cp.zeros((1, n_h), dtype=cp.float32)
    for item in seed: 
        x_embedded = self.params['W_embed'][item_to_idx[item]].reshape(1, -1)
        gates = x_embedded @ self.params['W_x'] + h @ self.params['W_h'] + self.params['b']
        h, c, _, _, _, _ = self._fused_forward_cell(gates[:,:n_h], gates[:,n_h:2*n_h], gates[:,2*n_h:3*n_h], gates[:,3*n_h:], c)
    res = list(seed)
    for _ in range(max_len):
        y = self.softmax(h @ self.params['Why'] + self.params['by']); idx = int(cp.argmax(y))
        if idx == end_idx: break
        res.append(idx_to_item[idx])
        x_embedded = self.params['W_embed'][idx].reshape(1, -1)
        gates = x_embedded @ self.params['W_x'] + h @ self.params['W_h'] + self.params['b']
        h, c, _, _, _, _ = self._fused_forward_cell(gates[:,:n_h], gates[:,n_h:2*n_h], gates[:,2*n_h:3*n_h], gates[:,3*n_h:], c)
    return "".join(res)

def create_batches_for_embedding(sequences, item_to_idx, batch_size):
    vocab_size = len(item_to_idx); indexed_sequences = [[item_to_idx[item] for item in seq] for seq in sequences]
    indexed_sequences.sort(key=len, reverse=True)
    batches = []
    for i in range(0, len(indexed_sequences), batch_size):
        batch_seqs = indexed_sequences[i:i+batch_size]; max_len = len(batch_seqs[0])
        X_batch_np = np.full((len(batch_seqs), max_len), 0, dtype=np.int32)
        Y_batch_np = np.full((len(batch_seqs), max_len, vocab_size), 0, dtype=np.float32)
        for j, seq in enumerate(batch_seqs):
            for k in range(len(seq) - 1): X_batch_np[j, k] = seq[k]; Y_batch_np[j, k, seq[k+1]] = 1
        batches.append((cp.array(X_batch_np), cp.array(Y_batch_np)))
    return batches
