# 최적화된 CNN 코드 설명 (Fully Optimized CNN)

---

## 1. CNN의 기본 개념

CNN은 이미지 등 2차원 데이터를 처리하는 딥러닝 모델로,
입력 이미지에서 **합성곱 필터**를 통해 특징 맵을 추출하고,
이를 여러 층 쌓아 고차원 표현으로 변환한 후,
MLP(다층 퍼셉트론) 등을 이용해 최종 분류 등을 수행한다.

---

## 2. 주요 하이퍼파라미터와 멤버 변수

* 입력 이미지 크기: $H \times W \times C$ (높이, 너비, 채널 수)
* 필터 크기: $F_H \times F_W$
* 스트라이드: $S$
* 패딩 크기: $P$
* 풀링 크기: $PoolSize$, 풀링 스트라이드: $PoolStride$
* 필터 개수: $num\_filters$
* 풀링 반복 횟수: $PoolingTimes$ (몇 층 합성곱+풀링 반복할지)
* 활성화 함수: ReLU (max(0,x))

---

## 3. CNN의 구조

입력 $X$에 대해,
각 층에서는 다음 순서로 진행된다.

$$
X \xrightarrow[\text{Convolution}]{\text{Conv Layer}} \text{Feature Map} \xrightarrow[\text{Activation}]{ReLU} \xrightarrow[\text{Batch Normalization}]{} \xrightarrow[\text{Pooling}]{} \text{다음 층 입력}
$$

마지막 층 풀링 후에는 1차원으로 펼쳐서 MLP에 넣어 최종 출력(분류 결과)을 얻는다.

---

## 4. 합성곱 연산 최적화

### 4.1 GEMM (General Matrix Multiply) 방식

합성곱 연산을 직접 계산하는 대신,
입력 이미지에서 슬라이딩 윈도우 형태의 패치를 모두 펼쳐서(im2col),
필터도 2차원 행렬로 펼친 후,
행렬 곱으로 처리한다.

즉,

* 입력 $X$를 크기 $(N, H, W, C)$에서 $(N, OH \times OW, F_H \times F_W \times C)$ 형태로 변환 (im2col)
* 필터를 크기 $(num\_filters, C, F_H, F_W)$에서 $(num\_filters, F_H \times F_W \times C)$로 변환
* 행렬곱으로 빠르게 합성곱 계산

---

### 4.2 Winograd 알고리즘

특히 $3 \times 3$ 필터, 스트라이드 1, 패딩 1 조건에서
Winograd 알고리즘(F(2x2, 3x3))을 이용해 합성곱을 계산하여 연산량을 크게 줄인다.

Winograd 행렬 $G, B, A$를 이용해 필터와 입력 타일을 변환하고,
소규모 행렬곱을 수행한 후 역변환한다.

---

## 5. 배치 단위 처리

입력 데이터를 배치(batch) 단위로 처리하여 GPU 병렬 연산 효율을 극대화한다.
`im2col_batch`, `PoolingLayer_batch`, `Flatten_batch` 함수들이 배치 연산에 최적화되어 있다.

---

## 6. 배치 정규화 (Batch Normalization)

합성곱 결과 Feature Map의 각 채널에 대해 평균과 분산을 계산하여 정규화한다.

$$
\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad y = \gamma \hat{x} + \beta
$$

여기서 $\gamma, \beta$는 학습 가능한 파라미터이며, Adam 옵티마이저를 통해 갱신된다. 학습 시에는 배치 통계(평균, 분산)를 사용하고, 추론 시에는 학습 중에 계산된 이동 평균(running mean)과 이동 분산(running variance)을 사용한다. 이를 위해 `is_training` 플래그를 활용한다.

---

## 7. 풀링 레이어

Max pooling 혹은 average pooling을 지원한다.

* 출력 크기:

$$
H_{out} = \left\lfloor \frac{H - PoolSize}{PoolStride} \right\rfloor + 1, \quad
W_{out} = \left\lfloor \frac{W - PoolSize}{PoolStride} \right\rfloor + 1
$$

* 각 풀링 영역에서 최대값 혹은 평균값을 취함

---

## 8. MLP (다층 퍼셉트론) 연결

최종 Feature Map을 1차원으로 펼쳐서(Flatten) MLP의 입력으로 사용한다.
MLP는 은닉층 크기를 자유롭게 지정 가능하며,
출력층은 분류 클래스 수에 대응한다.

---

## 9. 학습 절차

*   **순전파**: 배치 단위로 데이터를 읽어 합성곱-활성화-배치정규화-풀링을 반복해 Feature Map을 생성하고, 이를 MLP에 넣어 최종 출력값(분류 결과)을 계산한다.
*   **역전파 및 최적화**: 출력층의 오차를 기반으로 역전파를 수행하며, CNN 필터, 편향, 배치 정규화의 $\gamma, \beta$ 파라미터 및 MLP의 모든 가중치를 **Adam 옵티마이저**를 사용하여 갱신한다.
*   **학습률 스케줄링**: 5 에포크마다 학습률을 절반으로 감소시켜 학습의 안정성과 성능 향상을 도모한다.
*   **체크포인트 및 학습 재개**:
    *   `save_every` 파라미터에 따라 정기적으로 모델의 모든 파라미터(CNN, MLP 가중치, 배치 정규화 파라미터, Adam 옵티마이저의 모멘트 값, 현재 에포크)를 `.npz` 파일로 저장한다.
    *   `load_model` 함수를 통해 저장된 체크포인트로부터 학습을 정확히 재개할 수 있으며, 이때 Adam 옵티마이저의 상태도 함께 복원된다.
*   **조기 종료 (Early Stopping)**: `target_loss`에 도달하면 학습을 조기에 종료하여 과적합을 방지하고 효율적인 학습을 유도한다.

---

## 10. 가중치 초기화

Xavier 초기화 방식을 사용한다.

$$
std = \sqrt{\frac{2}{fan\_in + fan\_out}}
$$

여기서,

* $fan\_in = C \times F_H \times F_W$
* $fan\_out = num\_filters \times F_H \times F_W$

---

## 11. 주요 함수 요약

| 함수명                            | 역할                                       |
| ------------------------------ | ---------------------------------------- |
| `im2col_gpu` / `col2im_gpu`    | 입력을 행렬곱 가능한 형태로 변환 / 역변환               |
| `_convolution_forward`         | 합성곱 순전파 (im2col 기반)                       |
| `_batchnorm_forward`           | 배치 정규화 순전파 (학습 가능한 $\gamma, \beta$ 포함) |
| `_pooling_forward`             | 풀링 순전파                                   |
| `forward`                      | 전체 CNN 순전파 (MLP 포함)                      |
| `_convolution_backward`        | 합성곱 역전파                                   |
| `_batchnorm_backward`          | 배치 정규화 역전파 (dgamma, dbeta 계산)           |
| `_pooling_backward`            | 풀링 역전파                                   |
| `_update_params_adam`          | Adam 옵티마이저를 이용한 파라미터 업데이트             |
| `backward`                     | 전체 CNN 역전파 (MLP 포함)                      |
| `train`                        | 배치 단위 훈련, 학습률 스케줄링, 체크포인트 저장 및 재개 |
| `predict`                      | 예측 수행                                    |
| `save_model` / `load_model`    | 모델 저장/로드 (Adam 상태 및 에포크 포함)           |

---

## 12. 수식 및 출력 크기 계산

* 출력 Feature Map 크기 (합성곱 후):

$$
OH = \left\lfloor \frac{H + 2P - F_H}{S} \right\rfloor + 1, \quad
OW = \left\lfloor \frac{W + 2P - F_W}{S} \right\rfloor + 1
$$

* 풀링 후 크기:

$$
H' = \left\lfloor \frac{OH - PoolSize}{PoolStride} \right\rfloor + 1, \quad
W' = \left\lfloor \frac{OW - PoolSize}{PoolStride} \right\rfloor + 1
$$

---

## 13. 학습 관련 참고

*   입력 $x$와 정답 $y$를 받아 손실 함수 기준으로 가중치를 업데이트한다.
*   MLP 내부의 `front_propagation`, `back_propagation` 메서드를 사용하여 MLP 파라미터를 갱신한다.
*   학습률 스케줄링, 체크포인트 저장 및 로드, 조기 종료 기능이 통합되어 안정적이고 효율적인 훈련 환경을 제공한다.

---

# 결론

이 CNN 코드는 **Winograd, GEMM, 배치 처리, 다채널 지원, 풀링, 학습 가능한 배치 정규화, Adam 옵티마이저, MLP 연결, 학습률 스케줄링, 포괄적인 체크포인트 및 재개 기능, 조기 종료 기능까지 모두 갖춘 고도화된 합성곱 신경망 구현체**이다.

특히 3x3 필터일 때는 Winograd 알고리즘으로 빠르게 합성곱을 수행하며,
이를 실패 시 GEMM 방식으로 자동 전환하여 안정성도 확보하였다.
