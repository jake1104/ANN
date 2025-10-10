# 🧠 단층 퍼셉트론(SLP)의 개념과 구현

## 1. 퍼셉트론(Perceptron)의 개념과 역사

퍼셉트론은 1958년 프랭크 로젠블랫(Frank Rosenblatt)이 제안한 인공 뉴런 모델이다. 이는 생물학적 뉴런의 작동 원리를 수학적으로 모사한 모델로, 입력값들의 가중합을 기준으로 이진 출력을 내는 간단한 분류기이다.

퍼셉트론은 입력 벡터 x와 가중치 벡터 w의 내적과 편향 b의 합을 계산한 뒤, 그 결과에 활성화 함수(activation function)를 적용하여 출력을 결정한다. 초기에는 단순한 선형 분류 문제를 해결할 수 있었으나, XOR 문제처럼 선형적으로 분리되지 않는 문제를 해결할 수 없다는 한계 때문에 한동안 주목을 받지 못하였다. 이후 1986년 Rumelhart 등이 다층 퍼셉트론(MLP, Multi-Layer Perceptron)과 역전파 알고리즘을 발표하면서 다시 주목받게 되었다.

---

## 2. 단층 퍼셉트론(SLP)의 수학적 정의

단층 퍼셉트론은 다음과 같은 수식으로 정의된다:

$$
h = \sum_{i=1}^{n} w_i x_i + b = \mathbf{w}^\top \mathbf{x} + b
$$

$$
\hat{y} = f(h)
$$

여기서,

* $\mathbf{x} \in \mathbb{R}^n$: 입력 벡터
* $\mathbf{w} \in \mathbb{R}^n$: 가중치 벡터
* $b \in \mathbb{R}$: 편향(bias)
* $f$: 이진 계단 함수 (Binary Step Function)
* $\hat{y} \in \{0, 1\}$: 출력 값

이 때, 활성화 함수 $f(h)$는 다음과 같다:

$$
f(h) = \begin{cases}
1 & \text{if } h \geq 0 \\
0 & \text{otherwise}
\end{cases}
$$

---

## 3. 학습 알고리즘

퍼셉트론 학습 규칙은 오차 기반 업데이트 방식이다. 실제 출력 $y$와 예측값 $\hat{y}$의 차이를 바탕으로 가중치와 편향을 다음과 같이 갱신한다:

$$
\mathbf{w} \leftarrow \mathbf{w} + \eta (y - \hat{y}) \mathbf{x}
$$

$$
b \leftarrow b + \eta (y - \hat{y})
$$

여기서 $\eta$는 학습률(learning rate)이다.

---

## 4. 코드로 구현한 구조 해설

### 4.1 클래스 초기화

```python
class SLP:
  def __init__(self, n = 2):
    self.n : int = n
    self.ws : np.array = np.random.randn(n) / 100
    self.b : int = 0
```

* 입력 차원 수 $n$을 설정하고, 가중치 $\mathbf{w}$를 작은 난수로 초기화한다.
* 편향 $b$는 0으로 초기화한다.

---

### 4.2 예측 함수

```python
def predict(self, x):
  h = np.sum(np.dot(x, self.ws)) + self.b
  y = binary_step_func(h)
  return y
```

* $\mathbf{x}$와 $\mathbf{w}$의 내적 + 편향을 계산한 뒤, 계단 함수로 이진 출력을 반환한다.

---

### 4.3 학습 함수 (배치/온라인 방식)

```python
def classic_train(self, x, y, lr, epochs):
  for _ in range(epochs):
    e = y - self.predict(x)
    self.ws += lr * np.dot(e, x)
    self.b += lr * e
```

* classic\_train은 단일 샘플(x, y)에 대해서 반복하여 학습하는 형태이다.
* predict(x)를 호출해 예측값을 계산하고, 오차를 기반으로 가중치와 편향을 수정한다.

```python
def train(self, x, y, lr, epochs):
  for _ in range(epochs):
    for i in range(len(x)):
      e = y[i] - self.predict(x[i])
      self.ws = self.ws + lr * e * x[i]
      self.b += lr * e
```

* train은 미니배치가 아니라 샘플 단위로 순차적으로 업데이트한다 (온라인 학습 방식).

---

### 4.4 시각화

```python
plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_background)
plt.scatter(x[:, 0], x[:, 1], c=y, cmap=cmap_points, marker='o', s=100, label='Training Data')
```

* 결정 경계(decision boundary)를 시각화하여 학습된 퍼셉트론이 어떤 기준으로 분류를 수행하는지 확인할 수 있다.

---

## 5. AND, OR, NAND 문제 예시

퍼셉트론은 아래의 논리 연산들을 잘 분류할 수 있다:

| X₁ | X₂ | AND | OR | NAND |
| -- | -- | --- | -- | ---- |
| 0  | 0  | 0   | 0  | 1    |
| 0  | 1  | 0   | 1  | 1    |
| 1  | 0  | 0   | 1  | 1    |
| 1  | 1  | 1   | 1  | 0    |

그러나 XOR 문제는 선형 분리 불가능(linearly non-separable)이므로 단층 퍼셉트론으로는 해결할 수 없다. 이를 해결하기 위해서는 다층 퍼셉트론(MLP)이 필요하다.

---

## 6. 결론

단층 퍼셉트론은 가장 기초적인 형태의 인공신경망으로, 선형적으로 분리 가능한 문제를 빠르게 학습할 수 있다. 그러나 복잡한 문제를 다루기 위해서는 다층 구조 및 비선형 활성화 함수가 필수적이다. 이 구현은 인공신경망의 핵심 개념을 이해하기 위한 중요한 출발점이 된다.
