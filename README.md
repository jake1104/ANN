# Artificial Neural Networks (ANN) Implementations

This repository contains various implementations of Artificial Neural Networks (ANN) from fundamental concepts to more advanced architectures. The goal is to provide clear, concise, and functional examples of different neural network models.

## Implemented Models

Currently, the repository includes implementations for the following neural network architectures:

*   **SLP (Single Layer Perceptron)**: `_01_SLP/`
*   **MLP (Multi Layer Perceptron)**: `_02_MLP/`
    * **Example**: XOR
*   **CNN (Convolutional Neural Network)**: `_03_CNN/`
    * **Example**: MNIST
*   **RNN (Recurrent Neural Network)**: `_04_RNN/`
    *   Vanilla RNN: `_04_RNN/VanillaRNN.py`
        * **Example**: Word Completion
    *   LSTM (Long Short-Term Memory): `_04_RNN/LSTM/`
        * **Example**: Word Completion
    *   GRU (Gated Recurrent Unit): `_04_RNN/GRU/`
        * **Example**: Word Completion

Each model directory typically contains the implementation, examples, and relevant documentation.

## Future Plans

We plan to expand this repository to include:

*   **Transformer Models**: Implementation of attention-based architectures.
*   **Training Pipeline Construction**: Building robust and scalable training pipelines.
*   **Large Model Design**: Principles and practices for designing and implementing large-scale neural network models.

## How to use

- `py -m ANN.__init__`
