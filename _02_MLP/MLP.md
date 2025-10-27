# 🔗 MLP (Multi-Layer Perceptron) - Concepts and Implementation of Multi-Layer Perceptron

## 1. Overview

MLP (Multi-Layer Perceptron) is a neural network structure designed to overcome the limitations of Single Layer Perceptron (SLP), consisting of one or more hidden layers between the input and output layers. Through non-linear activation functions and a multi-layered structure, it can effectively solve non-linear problems. For example, MLP can solve the XOR problem, which cannot be solved by SLP.

---

## 2. MLP Structure and Equations

### 2.1 Layer Structure

MLP consists of layers of the following form:

* Input layer: $x \in \mathbb{R}^{n_0}$
* Hidden layer(s): $h^{(l)} \in \mathbb{R}^{n_l}$, $l = 1, \dots, L-1$
* Output layer: $\hat{y} \in \mathbb{R}^{n_L}$

### 2.2 Forward Propagation

The operation in hidden layer $l$ is defined as follows:

$$
z^{(l)} = a^{(l-1)} W^{(l)} + b^{(l)} \\
a^{(l)} = f(z^{(l)})
$$

Where,

* $W^{(l)} \in \mathbb{R}^{n_{l-1} \times n_l}$: Weight matrix
* $b^{(l)} \in \mathbb{R}^{1 \times n_l}$: Bias
* $f$: Activation function (sigmoid, etc.)
* $a^{(l)}$: Output of the l-th layer (input to the next layer)

The output layer uses the softmax function:

$$
\text{softmax}(z_j) = \frac{e^{z_j}}{\sum_{k=1}^{n} e^{z_k}}
$$

---

## 3. Loss Function

When the output layer is softmax and the ground truth is one-hot encoded, cross-entropy loss is used:

$$
\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{C} y_{ij} \log(\hat{y}_{ij})
$$

* $N$: Number of data samples
* $C$: Number of classes
* $y_{ij}$: Ground truth matrix
* $\hat{y}_{ij}$: Softmax probability output

---

## 4. Backpropagation

Backpropagation is the process of calculating the error of each layer and updating weights and biases based on it.

In reverse order from the output layer:

1. Output layer gradient:

$$
\delta^{(L)} = \hat{y} - y
$$

2. Hidden layer gradient:

$$
\delta^{(l)} = \left( \delta^{(l+1)} W^{(l+1)^\top} \right) \odot f'(z^{(l)})
$$

3. Weight, bias update:

$$
W^{(l)} \leftarrow \text{Adam}(W^{(l)}, \nabla_{W^{(l)}} \mathcal{L}) \\
b^{(l)} \leftarrow \text{Adam}(b^{(l)}, \nabla_{b^{(l)}} \mathcal{L})
$$

Here, $\odot$ denotes element-wise product (Hadamard product), and weights and biases are updated through the Adam optimizer.

### 4.4 Adam Optimizer

Adam (Adaptive Moment Estimation) is an optimization algorithm that adaptively adjusts the learning rate for each parameter. It performs updates by utilizing the first moment (mean) and second moment (variance) of previous gradients.

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
\hat{m}_t = m_t / (1 - \beta_1^t) \\
\hat{v}_t = v_t / (1 - \beta_2^t) \\
\theta_t = \theta_{t-1} - \alpha \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)
$$

Here, $g_t$ is the gradient at the current time step, $\alpha$ is the learning rate, $\beta_1, \beta_2$ are exponential weighted average coefficients, and $\epsilon$ is a small value to prevent the denominator from becoming zero.

---

## 5. Implementation Code Description

### 5.1 Class Initialization

```python
self.ws = []
self.bs = []
for i in range(len(layer_sizes) - 1):
  w = random_matrix(input_dim, output_dim)
  b = zeros((1, output_dim))
```

* Initializes weights and biases between each layer.
* Uses Xavier initialization to enhance learning stability.

---

### 5.2 Forward Propagation Implementation

```python
def front_propagation(self, x):
  A = [x]
  Z = []
  ...
  z = dot(a_prev, W) + b
  a = sigmoid(z) or softmax(z)
```

* Calculates z for each layer and passes it through an activation function to calculate a.
* The last layer outputs a probability distribution using softmax.

---

### 5.3 Backpropagation Implementation

```python
dz = pred - y
for l in reversed(range(len(self.ws))):
  dw = dot(A[l].T, dZ)
  dZ = (dA @ Wᵗ) * sigmoid'(z)
```

* Calculates the error of each layer starting from the output layer\'s error.
* Updates weights and biases based on the calculated error and the Adam optimizer.
* The `backward` method returns the gradient (`d_input`) for the input layer to be passed to higher layers (e.g., CNN).

---

### 5.4 Model Saving and Loading

```python
def save_model(self):
  np.savez("file.npz", w0=w0, b0=b0, w1=w1, b1=b1, ...)
```

* Converts cupy arrays to numpy arrays for saving.
* Can be restored later with load_model().
* When loading a model, logic is included to compare the saved `layer_sizes` with the current model\'s structure to check for inconsistencies, enhancing model loading stability.

---

### 5.5 Training Function (train_standalone)

The `train_standalone` function provides an independent training loop with the following characteristics:

*   **Mini-batch gradient descent**: Divides the entire dataset into small batches for training, accelerating learning speed and improving memory efficiency.
*   **Data shuffling**: Randomly shuffles data in each epoch to help the model generalize better without relying on data order.
*   **Early Stopping**: If the specified loss value (`target_loss` parameter) is reached, training is stopped early to prevent overfitting and reduce unnecessary computation.

---

### 5.6 Parameter Interface (get_parameters, set_parameters)

*   `get_parameters()`: Returns MLP weights and biases as a tuple list for compatibility with other models (e.g., CNN).
*   `set_parameters(params)`: Sets parameters (weights, biases) received from external sources to the MLP. This is useful when using MLP as part of a larger neural network structure.

## 6. Example: Solving the XOR Problem

```python
x = cp.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = cp.array([[1, 0], [0, 1], [0, 1], [1, 0]])  # One-hot encoding
```

The XOR problem cannot be solved linearly, so it cannot be solved by SLP, but MLP successfully learns it.

The prediction visualization results in a curved decision boundary as follows.

---

## 7. Visualization

*   Divides the input 2D space into a grid and visualizes the probability of Class 1 at each point as a contour plot.
*   Training data is displayed with colors and predicted values.

---

## 8. Conclusion

MLP is a fundamental yet powerful neural network structure. By including practical elements such as activation functions, loss functions, backpropagation, weight initialization, and model saving, it completes a learnable multi-layered structure.

The MLP_v04.py code has the following features:

*   ✅ GPU acceleration using Cupy
*   ✅ Flexible design of multi-layered structure
*   ✅ Softmax + cross entropy combination
*   ✅ Adam optimizer applied
*   ✅ Mini-batch training and data shuffling
*   ✅ Early Stopping function
*   ✅ Architecture verification during model saving and loading
*   ✅ Parameter interface with other models (get_parameters, set_parameters)
*   ✅ Capable of solving the XOR problem
*   ✅ Includes visualization