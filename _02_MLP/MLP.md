# 🔗 MLP (Multi-Layer Perceptron) - 다층 퍼셉트론의 개념과 구현

## 1. 개요

MLP(다층 퍼셉트론)는 단층 퍼셉트론(SLP)의 한계를 극복하기 위해 고안된 신경망 구조로, 입력층과 출력층 사이에 하나 이상의 은닉층(hidden layer)을 갖는 구조이다. 비선형 활성화 함수와 다층 구조를 통해 비선형적인 문제도 효과적으로 해결할 수 있다. 대표적으로 SLP로는 풀 수 없는 XOR 문제를 MLP로는 해결할 수 있다.

---

## 2. MLP의 구조 및 수식

### 2.1 레이어 구조

MLP는 다음과 같은 형태의 레이어로 구성된다:

* 입력층: $x \in \mathbb{R}^{n_0}$
* 은닉층(들): $h^{(l)} \in \mathbb{R}^{n_l}$, $l = 1, \dots, L-1$
* 출력층: $\hat{y} \in \mathbb{R}^{n_L}$

### 2.2 순전파 (Forward Propagation)

은닉층 $l$에서의 연산은 다음과 같이 정의된다:

$$
z^{(l)} = a^{(l-1)} W^{(l)} + b^{(l)} \\
a^{(l)} = f(z^{(l)})
$$

여기서,

* $W^{(l)} \in \mathbb{R}^{n_{l-1} \times n_l}$: 가중치 행렬
* $b^{(l)} \in \mathbb{R}^{1 \times n_l}$: 편향
* $f$: 활성화 함수 (sigmoid 등)
* $a^{(l)}$: l번째 층의 출력(다음 층의 입력)

출력층은 softmax 함수를 사용한다:

$$
\text{softmax}(z_j) = \frac{e^{z_j}}{\sum_{k=1}^{n} e^{z_k}}
$$

---

## 3. 손실 함수

출력층이 softmax이고 정답이 one-hot 인코딩인 경우, 교차 엔트로피 손실을 사용한다:

$$
\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{C} y_{ij} \log(\hat{y}_{ij})
$$

* $N$: 데이터 샘플 수
* $C$: 클래스 수
* $y_{ij}$: 정답 행렬
* $\hat{y}_{ij}$: softmax 확률 출력

---

## 4. 역전파 (Backpropagation)

역전파는 각 층의 오차를 계산하고, 이를 바탕으로 가중치와 편향을 업데이트하는 과정이다.

출력층부터 역순으로:

1. 출력층의 gradient:

$$
\delta^{(L)} = \hat{y} - y
$$

2. 은닉층의 gradient:

$$
\delta^{(l)} = \left( \delta^{(l+1)} W^{(l+1)^\top} \right) \odot f'(z^{(l)})
$$

3. 가중치, 편향 업데이트:

$$
W^{(l)} \leftarrow \text{Adam}(W^{(l)}, \nabla_{W^{(l)}} \mathcal{L}) \\
b^{(l)} \leftarrow \text{Adam}(b^{(l)}, \nabla_{b^{(l)}} \mathcal{L})
$$

여기서 $\odot$는 요소별 곱(Hadamard product)을 의미하고, 가중치와 편향은 Adam 옵티마이저를 통해 업데이트된다.

### 4.4 Adam 옵티마이저

Adam(Adaptive Moment Estimation)은 각 파라미터에 대해 적응적으로 학습률을 조정하는 최적화 알고리즘이다. 이전 그라디언트의 1차 모멘트(평균)와 2차 모멘트(분산)를 활용하여 업데이트를 수행한다.

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
\hat{m}_t = m_t / (1 - \beta_1^t) \\
\hat{v}_t = v_t / (1 - \beta_2^t) \\
\theta_t = \theta_{t-1} - \alpha \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)
$$

여기서 $g_t$는 현재 시점의 그라디언트, $\alpha$는 학습률, $\beta_1, \beta_2$는 지수 가중 평균 계수, $\epsilon$은 분모가 0이 되는 것을 방지하는 작은 값이다.

---

## 5. 구현 코드 설명

### 5.1 클래스 초기화

```python
self.ws = []
self.bs = []
for i in range(len(layer_sizes) - 1):
  w = random_matrix(input_dim, output_dim)
  b = zeros((1, output_dim))
```

* 각 레이어 간의 가중치와 편향을 초기화한다.
* Xavier 초기화를 사용해 학습 안정성을 높인다.

---

### 5.2 순전파 구현

```python
def front_propagation(self, x):
  A = [x]
  Z = []
  ...
  z = dot(a_prev, W) + b
  a = sigmoid(z) or softmax(z)
```

* 각 층에 대해 z를 계산하고 활성화 함수를 통과시켜 a를 계산한다.
* 마지막 층은 softmax를 사용하여 확률 분포를 출력한다.

---

### 5.3 역전파 구현

```python
dz = pred - y
for l in reversed(range(len(self.ws))):
  dw = dot(A[l].T, dZ)
  dZ = (dA @ Wᵗ) * sigmoid'(z)
```

* 출력층의 오차부터 시작해 각 층의 오차를 계산한다.
* 계산된 오차와 Adam 옵티마이저를 기반으로 가중치와 편향을 업데이트한다.
* `backward` 메서드는 상위 레이어(예: CNN)로 전달하기 위한 입력층에 대한 그라디언트(`d_input`)를 반환한다.

---

### 5.4 모델 저장 및 불러오기

```python
def save_model(self):
  np.savez("file.npz", w0=w0, b0=b0, w1=w1, b1=b1, ...)
```

* cupy 배열을 numpy 배열로 변환하여 저장한다.
* 추후 load\_model()에서 복원 가능
* `load_model` 시에는 저장된 `layer_sizes`와 현재 모델의 구조를 비교하여 불일치 여부를 확인하는 로직이 포함되어 모델 로딩의 안정성을 높인다.

---

### 5.5 학습 함수 (train_standalone)

`train_standalone` 함수는 다음과 같은 특징을 갖는 독립적인 학습 루프를 제공한다:

*   **미니배치 경사 하강법**: 전체 데이터셋을 작은 배치로 나누어 학습을 진행하여 학습 속도를 높이고 메모리 효율성을 개선한다.
*   **데이터 셔플링**: 각 에포크마다 데이터를 무작위로 섞어 모델이 데이터의 순서에 의존하지 않고 일반화 성능을 향상시키도록 돕는다.
*   **조기 종료 (Early Stopping)**: `target_loss` 파라미터를 통해 지정된 손실 값에 도달하면 학습을 조기에 중단하여 과적합을 방지하고 불필요한 계산을 줄인다.

---

### 5.6 파라미터 인터페이스 (get_parameters, set_parameters)

*   `get_parameters()`: MLP의 가중치와 편향을 다른 모델(예: CNN)과의 호환성을 위해 튜플 리스트 형태로 반환한다.
*   `set_parameters(params)`: 외부로부터 전달받은 파라미터(가중치, 편향)를 MLP에 설정한다. 이는 MLP를 더 큰 신경망 구조의 일부로 사용할 때 유용하다.

## 6. 예제: XOR 문제 해결

```python
x = cp.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = cp.array([[1, 0], [0, 1], [0, 1], [1, 0]])  # One-hot encoding
```

XOR 문제는 선형적으로 분리 불가능하므로 SLP로는 해결할 수 없지만, MLP는 이를 성공적으로 학습한다.

예측 시각화 결과는 다음과 같은 곡선적 결정 경계를 가진다.

---

## 7. 시각화

* 입력 2D 공간을 그리드로 나누고, 각 지점에서 Class 1의 확률을 예측하여 contour plot으로 시각화한다.
* 학습 데이터는 색상과 예측값으로 표시된다.

---

## 8. 결론

MLP는 신경망의 기본적이면서도 강력한 구조이다. 활성화 함수, 손실 함수, 역전파, weight initialization, 모델 저장 등 실용적인 요소들을 포함함으로써 학습 가능한 다층 구조를 완성한다.

MLP_v04.py 코드는 다음과 같은 특징을 갖는다:

* ✅ Cupy를 활용한 GPU 가속
* ✅ 다층 구조의 자유로운 설계
* ✅ softmax + cross entropy 조합
* ✅ Adam 옵티마이저 적용
* ✅ 미니배치 학습 및 데이터 셔플링
* ✅ 조기 종료 (Early Stopping) 기능
* ✅ 모델 저장 및 로드 시 아키텍처 검증
* ✅ 다른 모델과의 파라미터 인터페이스 (get_parameters, set_parameters)
* ✅ XOR 문제 해결 가능
* ✅ 시각화 포함
