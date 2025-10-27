# LSTM (Long Short-Term Memory)

### 1. Overview

LSTM is a recurrent neural network structure designed to solve the **long-term dependency problem** of traditional RNNs.
Vanilla RNNs transmit previous information through the hidden state $h_t$, but as sequences become longer, the **vanishing gradient problem** occurs, preventing distant past information from being properly reflected.

LSTM solves this problem by introducing a cell state $C_t$ and selectively remembering or forgetting information through a **gate structure**.

---

### 2. Components

The core of LSTM is its **gate structure**. Each gate determines how much information to pass through with a value between 0 and 1. Input data is converted into an embedding vector via `W_embed` and passed to the LSTM cell.

1. **Input Gate ($i_t$)**

   * Determines how much new information to reflect in the cell state based on the current input $x_t$ and the previous hidden state $h_{t-1}$.
   * Formula:
     $$
     i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)
     $$

2. **Forget Gate ($f_t$)**

   * Determines which information from the previous cell state ($C_{t-1}$) to forget.
   * Formula:
     $$
     f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)
     $$

3. **Cell Candidate State ($\tilde{C}_t$)**

   * Candidate information to be newly added.
   * Formula:
     $$
     \tilde{C}*t = \tanh(W_c x_t + U_c h*{t-1} + b_c)
     $$

4. **Cell State Update ($C_t$)**

   * Updated by reflecting the forget gate and input gate.
   * Formula:
     $$
     C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
     $$
   * Here, $\odot$ is element-wise multiplication.

5. **Output Gate ($o_t$)**

   * Determines the final hidden state $h_t$.
   * Formula:
     $$
     o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o)
     $$
     $$
     h_t = o_t \odot \tanh(C_t)
     $$

---

### 3. Advantages

*   **Capable of learning long-term dependencies**
    → Can remember information even if it's far in the past.
*   **Mitigates the vanishing gradient problem**
*   **Powerful for character sequences, speech, and time series data**

---

### 4. Vanilla RNN vs LSTM Comparison

| Item            | Vanilla RNN        | LSTM                                  |
| --------------- | ------------------ | ------------------------------------- |
| Memory unit     | Hidden state (h_t) | Hidden state (h_t) + Cell state (C_t) |
| Long-term dependency | Difficult          | Possible                              |
| Structure       | Simple             | Complex (4 gates)                     |
| Computation     | Low                | High                                  |

---

### 5. Implementation Details (CuPy Based)

*   **CuPy Fused Kernels**: Optimizes the performance of forward and backward propagation calculations using CuPy fused kernels such as `_fused_forward_cell` and `_fused_backward_cell`.
*   **Weight Initialization**: Xavier initialization is applied to all weights, including `W_embed`, `W_x`, `W_h`, `b`, `Why`, and `by`, to enhance learning stability.
*   **Adam Optimizer**: Model parameters (`W_embed`, `W_x`, `W_h`, `b`, `Why`, `by`) are updated through the Adam optimizer. Adam's moments (`m`, `v`) and time step (`t`) are managed with the model.
*   **Batch Processing**: `forward` and `backward` methods are implemented with batch processing to handle multiple sequences simultaneously, maximizing GPU utilization efficiency.
*   **During forward propagation**:
    1.  Input indices are converted into embedding vectors via `W_embed`.
    2.  Gate values ($i_t, f_t, o_t, g_t$) are calculated based on the input $x_t$ at each time step and the previous hidden state $h_{t-1}$.
    3.  The cell state $C_t$ and hidden state $h_t$ are updated using `_fused_forward_cell`.
    4.  The output $y_t = \text{softmax}(h_t W_{hy} + b_y)$ is calculated using the final hidden state $h_t$.
*   **During backward propagation**:
    1.  Gradients for each gate are calculated starting from the error of the output layer.
    2.  Backpropagation is performed, maintaining the gradient flow through the cell state $C_t$ using `_fused_backward_cell`.
    3.  **Gradient Clipping**: Clipping is applied to the calculated gradients to prevent exploding gradients.

---

### 6. Training & Prediction

*   **Training Data**: Batch-unit sequence data prepared through the `create_batches_for_embedding` function.
*   **Loss Function**: **Cross-entropy** is used to measure the difference between the model's predictions and the actual correct answers.
*   **Optimization**: The **Adam optimizer** is used to efficiently update all model parameters (embeddings, gate weights, output weights, etc.).
*   **Learning Rate Scheduling**: The learning rate is halved every 500 epochs to promote learning stability and performance improvement.
*   **Checkpoints and Learning Resumption**:
    *   All model parameters (weights, Adam optimizer's moment values, current epoch) are regularly saved to a `.npz` file according to the `save_every` parameter.
    *   Learning can be accurately resumed from a saved checkpoint via the `load_model` function, with the Adam optimizer's state also restored.
*   **Early Stopping**: If `target_loss` is reached, training is terminated early to prevent overfitting and induce efficient learning.
*   **Prediction**: The `predict` function sequentially generates the next character based on `seed_text`. It handles conversions between characters and indices using `item_to_idx` and `idx_to_item`, and controls prediction length and termination conditions via `max_len` and `end_idx`.

---

### 7. Practical Tips

*   Recommended hidden layer size (hidden_size) around 32~128.
*   Stabilize learning by slightly lowering the learning rate (lr) compared to Vanilla RNN.
*   GPU training possible after implementation based on CuPy/NumPy.
*   When learning multiple words, the **train_words()** pattern can be used as is.

---

### 8. Data Preparation (create_batches_for_embedding)

The `create_batches_for_embedding` function is a utility that efficiently prepares data for LSTM model training.

*   **Sequence Indexing**: Converts each item (character, etc.) in the input sequence into an integer index.
*   **Sorting by Length**: Sorts by sequence length for batch processing.
*   **Padding**: Pads all sequences within a batch to match the length of the longest sequence.
*   **One-Hot Encoding**: Converts the target sequence into one-hot encoded format.
*   **Batch Generation**: After the above process, generates CuPy array batches in the form of `(X_batch_idx, Y_batch)`.

---

### 9. Conclusion

LSTM is a powerful sequence model that effectively solves the long-term dependency problem of RNNs. The `LSTM_v05.py` implementation integrates the latest deep learning training techniques such as GPU acceleration using CuPy fused kernels, Adam optimizer, Xavier initialization, batch processing, learning rate scheduling, checkpoints and resumption, and early stopping, providing high performance and stability.