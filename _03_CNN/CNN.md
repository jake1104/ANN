# Optimized CNN Code Description (Fully Optimized CNN)

---

## 1. Basic Concepts of CNN

CNN is a deep learning model that processes 2D data such as images,
extracts feature maps through **convolutional filters** from input images,
stacks multiple layers to transform them into high-dimensional representations,
and then performs final classification using MLP (Multi-Layer Perceptron), etc.

---

## 2. Key Hyperparameters and Member Variables

* Input image size: $H \times W \times C$ (Height, Width, Number of Channels)
* Filter size: $F_H \times F_W$
* Stride: $S$
* Padding size: $P$
* Pooling size: $PoolSize$, Pooling stride: $PoolStride$
* Number of filters: $num\_filters$
* Number of pooling repetitions: $PoolingTimes$ (how many convolutional + pooling layers to repeat)
* Activation function: ReLU (max(0,x))

---

## 3. CNN Structure

For input $X$,
each layer proceeds in the following order.

$$
X \xrightarrow[\text{Convolution}]{\text{Conv Layer}} \text{Feature Map} \xrightarrow[\text{Activation}]{ReLU} \xrightarrow[\text{Batch Normalization}]{} \xrightarrow[\text{Pooling}]{} \text{Next Layer Input}
$$

After the last pooling layer, it is flattened into 1D and fed into an MLP to obtain the final output (classification result).

---

## 4. Convolutional Operation Optimization

### 4.1 GEMM (General Matrix Multiply) Method

Instead of directly calculating convolutional operations,
all sliding window-shaped patches from the input image are flattened (im2col),
the filter is also flattened into a 2D matrix,
and then processed by matrix multiplication.

That is,

* Convert input $X$ from size $(N, H, W, C)$ to $(N, OH \times OW, F_H \times F_W \times C)$ (im2col)
* Convert filter from size $(num\_filters, C, F_H, F_W)$ to $(num\_filters, F_H \times F_W \times C)$
* Quickly calculate convolution with matrix multiplication

---

### 4.2 Winograd Algorithm

Especially under the conditions of $3 \times 3$ filter, stride 1, and padding 1,
the Winograd algorithm (F(2x2, 3x3)) is used to calculate convolution, significantly reducing the number of operations.

Filter and input tiles are transformed using Winograd matrices $G, B, A$,
small-scale matrix multiplication is performed, and then inverse transformed.

---

## 5. Batch Processing

Input data is processed in batches to maximize GPU parallel processing efficiency.
`im2col_batch`, `PoolingLayer_batch`, `Flatten_batch` functions are optimized for batch operations.

---

## 6. Batch Normalization

Normalizes the convolutional result Feature Map by calculating the mean and variance for each channel.

$$
\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad y = \gamma \hat{x} + \beta
$$

Here, $\gamma, \beta$ are learnable parameters and are updated through the Adam optimizer. During training, batch statistics (mean, variance) are used, and during inference, running mean and running variance calculated during training are used. For this, the `is_training` flag is utilized.

---

## 7. Pooling Layer

Supports Max pooling or average pooling.

* Output size:

$$
H_{out} = \left\lfloor \frac{H - PoolSize}{PoolStride} \right\rfloor + 1, \quad
W_{out} = \left\lfloor \frac{W - PoolSize}{PoolStride} \right\rfloor + 1
$$

* Takes the maximum or average value in each pooling region

---

## 8. MLP (Multi-Layer Perceptron) Connection

The final Feature Map is flattened into 1D and used as input to the MLP.
The MLP can freely specify the hidden layer size,
and the output layer corresponds to the number of classification classes.

---

## 9. Training Procedure

*   **Forward Propagation**: Reads data in batches, repeats convolution-activation-batch normalization-pooling to generate Feature Maps, and feeds them into the MLP to calculate the final output value (classification result).
*   **Backpropagation and Optimization**: Performs backpropagation based on the error of the output layer, and updates CNN filters, biases, batch normalization's $\gamma, \beta$ parameters, and all MLP weights using the **Adam optimizer**.
*   **Learning Rate Scheduling**: The learning rate is halved every 5 epochs to promote learning stability and performance improvement.
*   **Checkpoints and Learning Resumption**:
    *   All model parameters (CNN, MLP weights, batch normalization parameters, Adam optimizer's moment values, current epoch) are regularly saved to a `.npz` file according to the `save_every` parameter.
    *   Learning can be accurately resumed from a saved checkpoint via the `load_model` function, with the Adam optimizer's state also restored.
*   **Early Stopping**: If `target_loss` is reached, training is terminated early to prevent overfitting and induce efficient learning.

---

## 10. Weight Initialization

Uses Xavier initialization method.

$$
std = \sqrt{\frac{2}{fan\_in + fan\_out}}
$$

Where,

* $fan\_in = C \times F_H \times F_W$
* $fan\_out = num\_filters \times F_H \times F_W$

---

## 11. Summary of Key Functions

| Function Name                 | Role                                       |
| ----------------------------- | ------------------------------------------ |
| `im2col_gpu` / `col2im_gpu`   | Converts input to matrix-multiplication-ready form / Inverse transform |
| `_convolution_forward`        | Convolutional forward propagation (im2col based) |
| `_batchnorm_forward`          | Batch normalization forward propagation (including learnable $\gamma, \beta$) |
| `_pooling_forward`            | Pooling forward propagation                |
| `forward`                     | Full CNN forward propagation (including MLP) |
| `_convolution_backward`       | Convolutional backward propagation         |
| `_batchnorm_backward`         | Batch normalization backward propagation (calculates dgamma, dbeta) |
| `_pooling_backward`           | Pooling backward propagation               |
| `_update_params_adam`         | Parameter update using Adam optimizer      |
| `backward`                    | Full CNN backward propagation (including MLP) |
| `train`                       | Batch-unit training, learning rate scheduling, checkpoint saving and resumption |
| `predict`                     | Performs prediction                        |
| `save_model` / `load_model`   | Model saving/loading (including Adam state and epoch) |

---

## 12. Formulas and Output Size Calculation

* Output Feature Map size (after convolution):

$$
OH = \left\lfloor \frac{H + 2P - F_H}{S} \right\rfloor + 1, \quad
OW = \left\lfloor \frac{W + 2P - F_W}{S} \right\rfloor + 1
$$

* Size after pooling:

$$
H' = \left\lfloor \frac{OH - PoolSize}{PoolStride} \right\rfloor + 1, \quad
W' = \left\lfloor \frac{OW - PoolSize}{PoolStride} \right\rfloor + 1
$$

---

## 13. Training Related Notes

*   Receives input $x$ and ground truth $y$ to update weights based on the loss function.
*   Updates MLP parameters using `front_propagation`, `back_propagation` methods within the MLP.
*   Learning rate scheduling, checkpoint saving and loading, and early stopping features are integrated to provide a stable and efficient training environment.

---

# Conclusion

This CNN code is a **highly advanced convolutional neural network implementation that includes Winograd, GEMM, batch processing, multi-channel support, pooling, learnable batch normalization, Adam optimizer, MLP connection, learning rate scheduling, comprehensive checkpoint and resumption capabilities, and early stopping functionality.**

Especially for 3x3 filters, convolution is performed quickly with the Winograd algorithm,
and stability is also ensured by automatically switching to the GEMM method if it fails.