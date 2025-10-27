# 🧠 Concepts and Implementation of Single Layer Perceptron (SLP)

## 1. Concepts and History of Perceptron

The perceptron is an artificial neuron model proposed by Frank Rosenblatt in 1958. It is a simple classifier that mathematically simulates the working principle of biological neurons, producing a binary output based on the weighted sum of input values.

The perceptron calculates the sum of the dot product of the input vector x and the weight vector w, plus the bias b, and then applies an activation function to the result to determine the output. Initially, it could solve simple linear classification problems, but it did not receive much attention for a while due to its limitation of not being able to solve non-linearly separable problems like the XOR problem. Later, in 1986, Rumelhart et al. published the Multi-Layer Perceptron (MLP) and the backpropagation algorithm, bringing it back into the spotlight.

---

## 2. Mathematical Definition of Single Layer Perceptron (SLP)

A single-layer perceptron is defined by the following equations:

$$
h = \sum_{i=1}^{n} w_i x_i + b = \mathbf{w}^\top \mathbf{x} + b
$$

$$$
\hat{y} = f(h)
$$$

Where,

* $\mathbf{x} \in \mathbb{R}^n$: Input vector
* $\mathbf{w} \in \mathbb{R}^n$: Weight vector
* $b \in \mathbb{R}$: Bias
* $f$: Binary Step Function
* $\hat{y} \in \{0, 1\}$: Output value

In this case, the activation function $f(h)$ is as follows:

$$
f(h) = \begin{cases}
1 & \text{if } h \geq 0 \\
0 & \text{otherwise}
\end{cases}
$$

---

## 3. Learning Algorithm

The perceptron learning rule is an error-based update method. Weights and biases are updated as follows based on the difference between the actual output $y$ and the predicted value $\hat{y}$:

$$
\mathbf{w} \leftarrow \mathbf{w} + \eta (y - \hat{y}) \mathbf{x}
$$

$$
b \leftarrow b + \eta (y - \hat{y})
$$

Here, $\eta$ is the learning rate.

---

## 4. Code Implementation Structure Explanation

### 4.1 Class Initialization

```python
class SLP:
  def __init__(self, n = 2):
    self.n : int = n
    self.ws : np.array = np.random.randn(n) / 100
    self.b : int = 0
```

* Sets the number of input dimensions $n$ and initializes weights $\mathbf{w}$ with small random numbers.
* Initializes bias $b$ to 0.

---

### 4.2 Prediction Function

```python
def predict(self, x):
  h = np.sum(np.dot(x, self.ws)) + self.b
  y = binary_step_func(h)
  return y
```

* Calculates the dot product of $\mathbf{x}$ and $\mathbf{w}$ + bias, then returns a binary output with a step function.

---

### 4.3 Training Function (Batch/Online Method)

```python
def classic_train(self, x, y, lr, epochs):
  for _ in range(epochs):
    e = y - self.predict(x)
    self.ws += lr * np.dot(e, x)
    self.b += lr * e
```

* classic_train is a form of iterative learning for a single sample (x, y).
* Calls predict(x) to calculate the predicted value and modifies weights and biases based on the error.

```python
def train(self, x, y, lr, epochs):
  for _ in range(epochs):
    for i in range(len(x)):
      e = y[i] - self.predict(x[i])
      self.ws = self.ws + lr * e * x[i]
      self.b += lr * e
```

* train updates sequentially per sample, not in mini-batches (online learning method).

---

### 4.4 Visualization

```python
plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_background)
plt.scatter(x[:, 0], x[:, 1], c=y, cmap=cmap_points, marker='o', s=100, label='Training Data')
```

* Visualizes the decision boundary to confirm how the trained perceptron performs classification.

---

## 5. AND, OR, NAND Problem Examples

Perceptrons can classify the following logical operations well:

| X₁ | X₂ | AND | OR | NAND |
| -- | -- | --- | -- | ---- |
| 0  | 0  | 0   | 0  | 1    |
| 0  | 1  | 0   | 1  | 1    |
| 1  | 0  | 0   | 1  | 1    |
| 1  | 1  | 1   | 1  | 0    |

However, the XOR problem is linearly non-separable, so it cannot be solved by a single-layer perceptron. A multi-layer perceptron (MLP) is needed to solve this.

---

## 6. Conclusion

The single-layer perceptron is the most basic form of artificial neural network and can quickly learn linearly separable problems. However, a multi-layered structure and non-linear activation functions are essential to handle complex problems. This implementation serves as an important starting point for understanding the core concepts of artificial neural networks.