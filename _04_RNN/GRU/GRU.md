# 🧠 GRU (Gated Recurrent Unit)

---

## 📘 1. Overview

**GRU** is a variant of RNN proposed by **Cho et al.** in 2014,
and similar to LSTM, it is a model designed to solve the **Long-Term Dependency problem**.

It has a **simpler structure** than LSTM, resulting in **faster learning speed**,
and is often used in many practical models due to its very similar performance.

---

## 🔍 2. Comparison with LSTM

| Item            | LSTM                | GRU               |
| --------------- | ------------------- | ----------------- |
| Number of gates | 3 (Input, Forget, Output gates) | 2 (Update, Reset gates) |
| Cell state $c_t$  | Exists              | None              |
| Hidden state $h_t$ | Exists              | Exists (Cell state = Hidden state) |
| Learning speed  | Slow                | Fast              |
| Structure complexity | High                | Low               |

➡️ GRU **does not have a separate cell state**, and all information is contained within the **hidden state $h_t$**.

---

## ⚙️ 3. Equation Summary

The operation of GRU is expressed by the following equations.

1️⃣ **Update gate**
Determines how much past information to retain.
$$
 z_t = \sigma(W_z x_t + U_z h_{t-1} + b_z)
$$

2️⃣ **Reset gate**
Determines how much past information to forget.
$$ 
 r_t = \sigma(W_r x_t + U_r h_{t-1} + b_r)
$$

3️⃣ **Candidate hidden state**
Calculates new information.
$$ 
 \tilde{h}*t = \tanh(W_h x_t + U_h (r_t \odot h*{t-1}) + b_h)
$$

4️⃣ **Hidden state update**
Combines the previous state and the new state to determine the final hidden state.
$$ 
 h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
$$

---

## 💻 4. GRU Python Implementation Example

This example shows a basic implementation of GRU based on `numpy`. For GPU acceleration, `cupy` can be used instead of `numpy`.

```python
# GRU (Gated Recurrent Unit)
# 2025-10-11: Vanilla GRU Implementation

import numpy as np

class VanillaGRU:
    def __init__(self, n_x, n_h, n_y, lr=0.01):
        self.n_x, self.n_h, self.n_y = n_x, n_h, n_y
        self.lr = lr

        # Initialize weights
        self.Wz = np.random.randn(n_x, n_h) * 0.01
        self.Uz = np.random.randn(n_h, n_h) * 0.01
        self.bz = np.zeros((1, n_h))

        self.Wr = np.random.randn(n_x, n_h) * 0.01
        self.Ur = np.random.randn(n_h, n_h) * 0.01
        self.br = np.zeros((1, n_h))

        self.Wh = np.random.randn(n_x, n_h) * 0.01
        self.Uh = np.random.randn(n_h, n_h) * 0.01
        self.bh = np.zeros((1, n_h))

        self.Why = np.random.randn(n_h, n_y) * 0.01
        self.by = np.zeros((1, n_y))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def softmax(self, x):
        exp = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)

    def forward(self, X):
        h = np.zeros((1, self.n_h))
        hs, ys = [], []

        for x_t in X:
            z = self.sigmoid(x_t @ self.Wz + h @ self.Uz + self.bz)
            r = self.sigmoid(x_t @ self.Wr + h @ self.Ur + self.br)
            h_tilde = np.tanh(x_t @ self.Wh + (r * h) @ self.Uh + self.bh)
            h = (1 - z) * h + z * h_tilde
            y = self.softmax(h @ self.Why + self.by)

            hs.append(h)
            ys.append(y)

        return np.array(hs), np.array(ys)

    def backward(self, X, Y, hs, ys):
        dWhy = np.zeros_like(self.Why)
        dby = np.zeros_like(self.by)
        dh_next = np.zeros((1, self.n_h))

        for t in reversed(range(len(X))):
            dy = ys[t] - Y[t]
            dWhy += hs[t].T @ dy
            dby += dy
            dh_next = dy @ self.Why.T + dh_next

        # For simplicity, this example omits Backprop calculations for each gate and only updates output layer weights.
        self.Why -= self.lr * np.clip(dWhy, -5, 5)
        self.by -= self.lr * np.clip(dby, -5, 5)

    def train(self, X, Y, epochs=2000):
        for epoch in range(epochs):
            hs, ys = self.forward(X)
            loss = -np.sum(Y * np.log(ys + 1e-8))
            self.backward(X, Y, hs, ys)
            if epoch % 200 == 0:
                print(f"Epoch {epoch}/{epochs} | Loss: {loss:.4f}")

    def predict(self, X):
        _, ys = self.forward(X)
        return np.argmax(ys, axis=2)

# Example: "hi" → "ih"
if __name__ == "__main__":
    vocab_size = 3
    X = np.eye(vocab_size)
    Y = np.roll(X, -1, axis=0)

    gru = VanillaGRU(n_x=vocab_size, n_h=8, n_y=vocab_size)
    gru.train(X, Y, epochs=1000)
```

---

## 🧩 5. Advantages of GRU

| Advantage        | Description                               |
| ---------------- | ----------------------------------------- |
| 🏃‍♂️ Fast learning | Fewer gates than LSTM, reducing computation |
| 💾 Less memory   | More efficient due to fewer parameters    |
| ⚖️ Similar performance | Capable of long-term dependency learning at LSTM level |
| 🔧 Simple structure | Easy to implement and tune                |

---

## 🧠 6. Key Application Areas

| Field            | Description                               |
| ---------------- | ----------------------------------------- |
| **Text generation** | Autocompletion, chatbot responses         |
| **Speech processing** | TTS (Text-to-Speech), speech emotion recognition |
| **Time series prediction** | Stock prices, weather, sensor data prediction |
| **Machine translation** | Utilized as Encoder-Decoder in Seq2Seq structure |

---

## 🌟 7. When is GRU good to use?

*   Where **real-time processing** is required (e.g., real-time translation, conversational AI)
*   Dealing with **small amounts of data or short sequences**
*   **Mobile/embedded environments** (when fast inference and low computation are important)

This implementation is `numpy`-based, but GPU acceleration can be achieved by utilizing `cupy` for faster training and inference.