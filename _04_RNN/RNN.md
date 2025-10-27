# 🔁 RNN (Recurrent Neural Network) - Concepts and Implementation of Recurrent Neural Networks

## 1. Overview

RNN (Recurrent Neural Network) is a neural network structure designed to process **sequence data**.
It can learn **temporal dependencies** between inputs and is typically used in **natural language processing, speech recognition, and time series prediction**.

While MLPs process inputs independently, RNNs have a recurrent structure by **reflecting the output of the previous time step into the input of the next time step**.

---

## 2. Structure and Equations

### 2.1 Basic Structure

An RNN consists of input $x_t$, hidden state $h_t$, and output $y_t$ at time $t$.

* Input: $x_t \in \mathbb{R}^{n_x}$
* Hidden state: $h_t \in \mathbb{R}^{n_h}$
* Output: $y_t \in \mathbb{R}^{n_y}$

The time-step recurrent structure is as follows:

```
x_t ─▶ [RNN Cell] ─▶ h_t ─▶ y_t
           ▲
           │
          h_{t-1}
```

---

### 2.2 Forward Propagation

The core equations of RNN are as follows:

$$
h_t = f(W_{xh} x_t + W_{hh} h_{t-1} + b_h)
$$

$$ 
y_t = g(W_{hy} h_t + b_y)
$$ 

Where,

* $W_{xh} \in \mathbb{R}^{n_x \times n_h}$ : Input → Hidden weight
* $W_{hh} \in \mathbb{R}^{n_h \times n_h}$ : Previous hidden → Current hidden weight
* $W_{hy} \in \mathbb{R}^{n_h \times n_y}$ : Hidden → Output weight
* $b_h$, $b_y$ : Biases
* $f$ : tanh (Hidden activation function)
* $g$ : softmax (Output activation function)

---

### 2.3 Unrolled Form

Unrolled along the time axis, it takes the following form:

$$ 
\begin{align*}
h_1 &= f(W_{xh} x_1 + W_{hh} h_0 + b_h) \\
h_2 &= f(W_{xh} x_2 + W_{hh} h_1 + b_h) \\
\vdots \\
h_T &= f(W_{xh} x_T + W_{hh} h_{T-1} + b_h)
\end{align*} 
$$ 

The output at each time step is:

$$ 
y_t = g(W_{hy} h_t + b_y)
$$ 

---

## 3. Loss Function

The total loss of the sequence is calculated as the average of the losses at all time steps.

$$ 
\mathcal{L} = -\frac{1}{T} \sum_{t=1}^{T} \sum_{j=1}^{C} y_{tj} \log(\hat{y}_{tj})
$$ 

---

## 4. Backpropagation (BPTT: Backpropagation Through Time)

RNN learning is performed using the **Backpropagation Through Time (BPTT)** algorithm.
The basic idea is to extend the backpropagation of MLP along the "time axis".

### 4.1 Basic Equations

Output layer error:

$$ 
\delta^{(y)}_t = \hat{y}_t - y_t
$$ 

Hidden layer error (backpropagation):

$$ 
\delta^{(h)}*t = (\delta^{(y)}*t W*{hy}^\top + \delta^{(h)}*{t+1} W_{hh}^\top) \odot f'(h_t)
$$ 

Here, $\odot$ is the element-wise product (Hadamard product).

### 4.2 Gradient Accumulation

Gradients from all time steps are accumulated and used for weight updates, and gradient clipping is applied to prevent exploding gradients:

$$ 
\frac{\partial \mathcal{L}}{\partial W_{xh}} = \sum_{t=1}^{T} x_t^\top \delta^{(h)}*t \\
\frac{\partial \mathcal{L}}{\partial W_{hh}} = \sum_{t=1}^{T} h_{t-1}^\top \delta^{(h)}*t \\
\frac{\partial \mathcal{L}}{\partial W_{hy}} = \sum_{t=1}^{T} h_t^\top \delta^{(y)}_t
$$ 

---

## 5. Implementation Code Description

### 5.1 Class Initialization

```python
class VanillaRNN:
    def __init__(self, n_x, n_h, n_y, lr=0.01):
        self.Wxh = np.random.randn(n_x, n_h) * 0.01
        self.Whh = np.random.randn(n_h, n_h) * 0.01
        self.Why = np.random.randn(n_h, n_y) * 0.01
        self.bh = np.zeros((1, n_h))
        self.by = np.zeros((1, n_y))
        self.lr = lr
```

---

### 5.2 Forward Pass

```python
def forward(self, X):
    h, hs, ys = np.zeros((1, self.Whh.shape[0])), [], []
    for x_t in X:
        h = np.tanh(x_t @ self.Wxh + h @ self.Whh + self.bh)
        y = self.softmax(h @ self.Why + self.by)
        hs.append(h)
        ys.append(y)
    return np.array(hs), np.array(ys)
```

---

### 5.3 Backpropagation (BPTT)

```python
def backward(self, X, Y, hs, ys):
    dWxh = np.zeros_like(self.Wxh)
    dWhh = np.zeros_like(self.Whh)
    dWhy = np.zeros_like(self.Why)
    dbh = np.zeros_like(self.bh)
    dby = np.zeros_like(self.by)

    dh_next = np.zeros_like(hs[0])

    for t in reversed(range(len(X))):
        dy = ys[t] - Y[t]
        dWhy += hs[t].T @ dy
        dby += dy
        dh = dy @ self.Why.T + dh_next
        dh_raw = (1 - hs[t]**2) * dh
        dbh += dh_raw
        dWxh += X[t].T @ dh_raw
        dWhh += hs[t-1].T @ dh_raw if t > 0 else 0
        dh_next = dh_raw @ self.Whh.T

    # Update
    for param, dparam in zip(
        [self.Wxh, self.Whh, self.Why, self.bh, self.by],
        [dWxh, dWhh, dWhy, dbh, dby]
    ):
        param -= self.lr * np.clip(dparam, -5, 5)
```

---

### 5.4 Learning and Prediction Utility Functions

*   `compute_loss(Y, ys)`: Calculates cross-entropy loss based on actual answers `Y` and model predictions `ys`.
*   `prepare_sequences(words, char_to_idx)`: Prepares training data (`X_train`, `Y_train`) by converting a given list of words into character-level one-hot encoded sequences.
*   `train_words(words, char_to_idx, epochs, print_interval)`: A convenient wrapper function that prepares training data by calling `prepare_sequences` and then trains the model using the `train` method.
*   `predict(seed_text, char_to_idx, idx_to_char, length)`: Generates a sequence by predicting the next character `length` times using the RNN model, starting with the given `seed_text`. During prediction, the previously predicted character is used as the input for the next time step.

## 6. Example: Character-Level RNN

```python
# Input: ["h", "e", "l", "l", "o"]
# Output: ["e", "l", "l", "o", " "]
X = np.eye(5)  # One-hot encoding
Y = np.roll(X, -1, axis=0)

rnn = VanillaRNN(n_x=5, n_h=8, n_y=5)

for epoch in range(1000):
    hs, ys = rnn.forward(X)
    loss = -np.sum(Y * np.log(ys + 1e-8))
    rnn.backward(X, Y, hs, ys)
```

---

## 7. Visualization

*   Visualizing changes in hidden states with t-SNE, etc., allows observation of **contextual patterns within sequences**.
*   Representing output probabilities as a heatmap provides an intuitive understanding of the model's "prediction distribution".

---

## 8. Conclusion

Vanilla RNN is the most basic recurrent neural network and supports GPU acceleration using `cupy`.
While it can process sequence data, it has difficulty remembering long contexts due to the **long-term dependency problem**.

To overcome this limitation, **LSTM** and **GRU** emerged later.

---

### 🔗 Next Steps

| Model        | Features                                  |
| ------------ | ----------------------------------------- |
| LSTM         | Solves long-term dependency with gate structure |
| GRU          | Similar performance with simpler structure than LSTM |
| BiRNN        | Learns bidirectional contextual information |
| Seq2Seq      | Encoder-Decoder structure, applied to translation/summarization |
| Attention    | Selective information focusing (basis of Transformer) |