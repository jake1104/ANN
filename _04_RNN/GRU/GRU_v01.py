
# Created at
# 2025-10-11: Created GRU model based on LSTM_v05.py

# Features
# - GRU (Gated Recurrent Unit) Layer
# - Embedding Layer
# - Adam Optimizer
# - Softmax Output Layer
# - Loss Calculation (Cross-Entropy)
# - Batch Training
# - Learning Rate Scheduler
# - Checkpointing (Save/Load Model)
# - Text Generation (Prediction)
# - Fused CUDA kernels for performance (cp.fuse())

import cupy as cp
import numpy as np
import os

class GRU:
  @staticmethod
  def _fused_forward_cell(x_proj, h_proj, b, h_prev, n_h):
    g_r = x_proj[:, :n_h] + h_proj[:, :n_h] + b[:, :n_h]
    g_z = x_proj[:, n_h:2*n_h] + h_proj[:, n_h:2*n_h] + b[:, n_h:2*n_h]
    
    r_t = 1 / (1 + cp.exp(-g_r))
    z_t = 1 / (1 + cp.exp(-g_z))
    
    g_n = x_proj[:, 2*n_h:] + r_t * h_proj[:, 2*n_h:] + b[:, 2*n_h:]
    n_t = cp.tanh(g_n)
    
    h_t = (1 - z_t) * h_prev + z_t * n_t
    return h_t, r_t, z_t, n_t, h_proj[:, 2*n_h:]

  @staticmethod
  def _fused_backward_cell(dh, W_h, h_prev, r_t, z_t, n_t, h_proj_n, n_h):
    W_hr, W_hz, W_hn = W_h[:, :n_h], W_h[:, n_h:2*n_h], W_h[:, 2*n_h:]

    # Backprop through h_t = (1 - z_t) * h_prev + z_t * n_t
    dn_t = dh * z_t
    dh_prev_from_ht = dh * (1 - z_t)
    dz_t = dh * (n_t - h_prev)

    # Backprop through n_t = tanh(g_n)
    dg_n = dn_t * (1 - n_t**2)

    # Backprop through g_n = x_n + r_t * h_proj_n + b_n
    dr_t_from_gn = dg_n * h_proj_n
    dh_proj_n_from_gn = dg_n * r_t
    dh_prev_from_hpn = dh_proj_n_from_gn @ W_hn.T

    # Backprop through z_t = sigmoid(g_z)
    dg_z = dz_t * z_t * (1 - z_t)
    dh_prev_from_hpz = dg_z @ W_hz.T

    # Backprop through r_t = sigmoid(g_r)
    dg_r = dr_t_from_gn * r_t * (1 - r_t)
    dh_prev_from_hpr = dg_r @ W_hr.T

    # Combine gradients for h_prev
    dh_next = dh_prev_from_ht + dh_prev_from_hpn + dh_prev_from_hpz + dh_prev_from_hpr

    return dg_r, dg_z, dg_n, dh_next

  def __init__(self, vocab_size, embedding_dim, n_h, n_y):
    # Xavier Init for GRU (3 gates)
    limit_embed = np.sqrt(6.0 / (vocab_size + embedding_dim))
    limit_wx = np.sqrt(6.0 / (embedding_dim + 3 * n_h))
    limit_wh = np.sqrt(6.0 / (n_h + 3 * n_h))
    limit_why = np.sqrt(6.0 / (n_h + n_y))

    # Model Parameters
    self.params = {
        'W_embed': cp.random.uniform(-limit_embed, limit_embed, (vocab_size, embedding_dim), dtype=cp.float32),
        'W_x': cp.random.uniform(-limit_wx, limit_wx, (embedding_dim, 3 * n_h), dtype=cp.float32),
        'W_h': cp.random.uniform(-limit_wh, limit_wh, (n_h, 3 * n_h), dtype=cp.float32),
        'b': cp.zeros((1, 3 * n_h), dtype=cp.float32),
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
    n_h = self.params['W_h'].shape[0]
    X_embedded = self.params['W_embed'][X_batch_idx]
    hs = cp.zeros((batch_size, seq_len, n_h), dtype=cp.float32)
    ys = cp.zeros((batch_size, seq_len, self.params['Why'].shape[1]), dtype=cp.float32)
    caches = []
    h_prev = cp.zeros((batch_size, n_h), dtype=cp.float32)

    for t in range(seq_len):
      x_t = X_embedded[:, t, :]
      x_proj = x_t @ self.params['W_x']
      h_proj = h_prev @ self.params['W_h']
      
      h_t, r_t, z_t, n_t, h_proj_n = self._fused_forward_cell(x_proj, h_proj, self.params['b'], h_prev, n_h)
      hs[:, t, :] = h_t # Store hidden state
      ys[:, t, :] = self.softmax(h_t @ self.params['Why'] + self.params['by'])
      caches.append((x_t, h_prev, r_t, z_t, n_t, h_proj_n, h_t))
      h_prev = h_t
      
    caches.append(X_batch_idx)
    return hs, ys, caches

  def backward(self, Y_batch, ys, caches):
    batch_size, seq_len, _ = Y_batch.shape
    X_batch_idx = caches.pop()
    n_h, n_y, embedding_dim = self.params['W_h'].shape[0], self.params['Why'].shape[1], self.params['W_embed'].shape[1]
    
    grads = {p: cp.zeros_like(self.params[p]) for p in self.params}
    dh_next = cp.zeros((batch_size, n_h), dtype=cp.float32)
    dX_embedded = cp.zeros((batch_size, seq_len, embedding_dim), dtype=cp.float32)
    dy = ys - Y_batch
    
    for t in reversed(range(seq_len)):
      dy_t = dy[:, t, :]
      x_t, h_prev, r_t, z_t, n_t, h_proj_n, h_t = caches[t]
      
      grads['Why'] += h_t.T @ dy_t
      grads['by'] += cp.sum(dy_t, axis=0, keepdims=True)
      
      dh = dy_t @ self.params['Why'].T + dh_next
      
      dg_r, dg_z, dg_n, dh_next = self._fused_backward_cell(dh, self.params['W_h'], h_prev, r_t, z_t, n_t, h_proj_n, n_h)
      
      # Gradients for biases
      grads['b'][:, :n_h] += cp.sum(dg_r, axis=0)
      grads['b'][:, n_h:2*n_h] += cp.sum(dg_z, axis=0)
      grads['b'][:, 2*n_h:] += cp.sum(dg_n, axis=0)
      
      # Gradients for W_x
      dgates_x = cp.hstack((dg_r, dg_z, dg_n))
      grads['W_x'] += x_t.T @ dgates_x
      
      # Gradients for dX_embedded
      dX_embedded[:, t, :] = dgates_x @ self.params['W_x'].T
      
      # Gradients for W_h
      dh_proj_n_from_gn = dg_n * r_t
      grads['W_h'][:, :n_h] += h_prev.T @ dg_r
      grads['W_h'][:, n_h:2*n_h] += h_prev.T @ dg_z
      grads['W_h'][:, 2*n_h:] += h_prev.T @ dh_proj_n_from_gn
      
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

  def train(self, batches, epochs, lr, print_interval=100, save_every=500, checkpoint_dir="checkpoints", model_name="gru_model", start_epoch=0, target_loss=None):
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
    n_h = self.params['W_h'].shape[0]
    h = cp.zeros((1, n_h), dtype=cp.float32)
    
    # "Warm up" the hidden state with the seed sequence
    for item in seed: 
        x_idx = item_to_idx[item]
        x_embedded = self.params['W_embed'][x_idx].reshape(1, -1)
        
        # GRU forward step
        x_proj = x_embedded @ self.params['W_x']
        h_proj = h @ self.params['W_h']
        g_r = x_proj[:, :n_h] + h_proj[:, :n_h] + self.params['b'][:, :n_h]
        g_z = x_proj[:, n_h:2*n_h] + h_proj[:, n_h:2*n_h] + self.params['b'][:, n_h:2*n_h]
        r = self.sigmoid(g_r)
        z = self.sigmoid(g_z)
        g_n = x_proj[:, 2*n_h:] + r * h_proj[:, 2*n_h:] + self.params['b'][:, 2*n_h:]
        n = cp.tanh(g_n)
        h = (1 - z) * h + z * n

    res = list(seed)
    for _ in range(max_len):
        # Predict next character
        y = self.softmax(h @ self.params['Why'] + self.params['by'])
        idx = int(cp.argmax(y))
        
        if idx == end_idx: break
        res.append(idx_to_item[idx])
        
        # Update hidden state with the predicted character
        x_embedded = self.params['W_embed'][idx].reshape(1, -1)
        x_proj = x_embedded @ self.params['W_x']
        h_proj = h @ self.params['W_h']
        g_r = x_proj[:, :n_h] + h_proj[:, :n_h] + self.params['b'][:, :n_h]
        g_z = x_proj[:, n_h:2*n_h] + h_proj[:, n_h:2*n_h] + self.params['b'][:, n_h:2*n_h]
        r = self.sigmoid(g_r)
        z = self.sigmoid(g_z)
        g_n = x_proj[:, 2*n_h:] + r * h_proj[:, 2*n_h:] + self.params['b'][:, 2*n_h:]
        n = cp.tanh(g_n)
        h = (1 - z) * h + z * n
        
    return "".join(res)

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
        batches.append((cp.array(X_batch_np), cp.array(Y_batch_np)))
    return batches