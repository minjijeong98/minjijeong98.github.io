---
title: 데이터 전처리 - 정규화, 표준화
description: 정규화, 표준화의 의미와 사용
author: mj
date: 2024-08-22 15:30:00 +0900
categories: [EDA & Visualization]
tags: [boostcamp, preprocessing]
use_math: true
pin: false
---

# Outline

- **목적**: 딥러닝 뿐만 아니라 데이터를 다루는 거의 모든 상황에서 정규화 또는 표준화 개념이 자주 등장한다. 데이터의 특징 및 사용 모델에 따라 왜, 언제 이런 scaling을 적용해야 하는지 궁금해 살펴보려 한다.
- **cue**: 부스트캠프의 이번 주차 EDA 강의 중의 한 further question
- > "대부분의 경우 모델 학습에 긍정적인 영향을 줍니다. 정규화와 표준화의 경우 데이터를 특정 범위로 제한하게 되는데, 모델 학습에 부정적인 영향을 주는 경우가 있을까요? 그렇다면 어떤 경우에 모델에 부정적인 영향을 주게 될까요?"


# 1. 정규화(normalization) 및 표준화(standardiization)의 개념

- 일반적으로 데이터에서 feature들에 대해서만 정규화 및 표준화를 고려하며, target value에 대한 scaling은 요구되지 않는다. 따라서 본 글에서도 feature를 기준으로 정규화 및 표준화를 살펴본다.
- feature 간의 scale에 상당한 차이가 있거나 feature의 scale에 민감한 알고리즘을 사용하는데 이 scale을 조정해두지 않으면, 원활한 모델 학습을 저해할 수 있다. 정규화와 표준화는 이러한 문제를 방지해 모델 성능을 끌어올리기 위한 전략으로 이해할 수 있다.

|           | **정규화** (normalization)                                                                                                                                                                                                   | **표준화** (standardization)                                                                                                                                         |
| --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 정의      | - 데이터를 특정 범위로 변환해 범위를 일치시키는 작업<br>- 일반적으로 min-max normalization 의미                                                                                                                              | - 평균을 0으로, 표준편차를 1로 변환해 데이터를 조정하는 작업<br>- 일반적으로 Z-score standardization 의미                                                            |
| 수식      | - $[0,1]$ 범위 변환: $x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}}$<br>- $[a,b]$ 범위 변환: $x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}} \cdot (b-a) + a$                                                       |
| 특징      | - 데이터에 이상치가 없으며 분포가 크게 치우쳐 있지 않을 때 유용<br>- 이상치에 민감<br>- 범위가 제한됨                                                                                                                        | - 데이터에 이상치가 있거나, 분포가 치우쳐 있을 때 유용<br>- 범위 제한되지 않음. $(-\infty, \infty)$                                                                  |
| 단점      | 이상치에 민감함                                                                                                                                                                                                              |                                                                                                                                                                      |
| 적용 case | - feature의 범위가 다른 경우 (예: $[0,255]$ 범위의 이미지 픽셀값을 $[0,1]$ 범위로 조정)<br>- feature들의 scale이 다양한데, feature의 분포를 모르거나 / 모델에서 분포 가정을 하지 않거나 / 정규분포를 따르지 않음을 아는 경우 | - feature가 정규분포를 따르는 경우<br>- feature들의 scale이 다양한데, 모델이 feature에 대해 정규분포를 가정하고 있는 경우 (예: 선형회귀, 로지스틱 회귀, LDA, SVM 등) |
| python    | scikit-learn의 `MinMaxScaler`                                                                                                                                                                                                | scikit-learn의 `StandardScaler`                                                                                                                                      |


# 2. 정규화, 표준화 이유

### 1. 모델 편향 감소

- ML 모델은 feature 간 scale 차이로 인해 올바른 학습을 수행하지 못할 수도 있다. 일반적으로 scale이 큰 feature의 영향이 모델 학습에서 비대해지곤 한다. 이러한 문제를 정규화/표준화를 통해 예방한다.
- 거리 계산에 의존하는 모델은 더 큰 범위의 feature에 대해 편향될 수 있다. 정규화를 통해 모든 feature가 거리 metric에 동등하게 기여하도록 보장할 수 있다.
- 딥러닝에서는 '초반 학습에서 일부 feature의 영향이 강한 문제가 있다 하더라도, 학습 과정에서 모델이 알아서 해당 feature에 작은 weight을 할당하는 등의 방식으로 중요성을 구분할 수 있지 않을까?'라는 질문이 있을 수 있다. 
	- 그럴 수도 있지만, scaling 없이 학습하면 학습되는 속도가 상당히 오래 걸리므로 비효율적이다. 굳이 그런 비용을 치를 필요는 없다.
	- 또한 큰 feature 값은 계산에서 수치적 불안정성을 초래할 수 있으므로 정규화가 필요하다.

### 2. 모델의 수렴 속도 향상

- gradient descent 등의 방법론을 사용할 때, 정규화/표준화된 데이터를 사용하면 수렴 속도가 향상된다. 이는 feature들에 대한 step size가 어느정도 통일되며 최적화 환경이 보다 균일해지기 떄문으로 이해할 수 있다.
- 데이터의 범위를 통일함으로써 데이터의 불필요한 차원을 줄여, 모델 학습이 효율적으로 진행되도록 할 수 있다.
- 신경망 모델에서, 정규화되지 않은 큰 값을 사용하면 local minima에 빠질 위험이 있다. 정규화/표준화를 통해 그 가능성을 감소시킬 수 있다.

### 3. 수치적 불안정성 방지

- 큰 feature 값은 계산에서 수치적 불안정성을 초래할 수 있으며, 특히 행렬 연산을 포함하는 알고리즘에서 그렇다. 정규화를 통해 이 문제를 완화할 수 있다. 

### 4. 해석성 향상

- feature를 표준화하면 모델 coefficient를 더 해석하기 쉽게 만들 수 있다.
- 특히 선형 회귀식에서 feature 간의 직접적인 비교가 가능해지고, feature가 target에 미치는 영향을 직관적으로 설명 가능해진다.
- 예: "$x$가 1 표준편차만큼 증가/감소하면 $y$가 계수만큼 증가/감소한다"

# 3. 언제 정규화, 표준화를 적용할까?

### feature의 scale에 민감한 알고리즘

- gradient descent 기반 알고리즘(선형회귀, 로지스틱 회귀, 신경망, PCA 등): 모델의 수렴 속도와 관련해서, 정규화 및 표준화가 중요하다. 특히 신경망은 보통 입력 데이터의 크기와 분포에 민감하게 반응하므로 (비교적 큰 값을 가지면 상대적으로 최적화하기 어려움), 정규화를 통해 데이터의 범위를 $[0,1]$ 또는 $[-1,1]$ 로 조정하는 것이 일반적이다.
- 거리 기반 알고리즘 (kNN, K-means clustering, SVM 등): 데이터 간의 거리를 이용해 데이터의 유사성을 결정하므로, feature scale의 영향을 많이 받는다. 정규화를 통해 모든 feature가 output에 동등하게 기여하도록 조정해야 한다.

### 정규화, 표준화가 필요 없는 경우

- 트리 기반 모델 (의사결정 트리, 랜덤 포레스트 등)
	- 데이터에 특정 질문을 하는 "If-then" 원칙에 따라 작동하므로 데이터의 상대적인 순서를 고려하지, 데이터의 scaling에는 크게 영향 받지 않는다. 
	- 단지 데이터를 분할할 지점만 찾기 때문에, 이러한 알고리즘에서 스케일을 변경해도 성능이나 정확도가 향상되지 않는다.
- 모든 feature가 동일한 측정 단위로 존재하는 경우
	- 예를 들어, 지난 중간고사와 기말고사 영어 성적을 바탕으로 1반과 2반 중 영어 성적이 더 높은 반을 예측하는 ML 모델이라고 하면, 지난 중간/기말고사라는 두 feature는 $[0,100]$ 이라는 동일한 척도에 존재한다. 이런 경우 정규화, 표준화가 불필요하다.
- Naive Bayes 알고리즘과 같은 확률론적 모델
	- Naive Bayes 알고리즘은 사전확률과 사후확률만으로 작동한다. 사건의 발생 여부와 빈도만을 고려하므로, 데이터의 scale은 해당 알고리즘에서 주요 요인이 아니고, 따라서 성능에 영향을 미치지 않는다.
- ARIMA, SARIMA 등 auto-regressive 알고리즘

### 정규화와 표준화 중 뭘 적용할까?

- scaling은 모델 성능 향상에 크게 도움이 되므로, 정규화든 표준화든, 본격적인 분석 전의 필수 과정으로 생각하는게 좋다. 다만 각 feature들이 어떤 분포를 가질 때, 어떤 scaling을 적용하면 좋다는 정해진 공식은 없다. 따라서 풀고자 하는 문제 상황, 데이터의 특성, 모델의 가정을 확인하고, 정규화 또는 표준화 적용 전후 결과를 비교하며 scaling을 경험적으로 적용하는 것이 일반적이다. 

### 이상치는 언제 제거할까?

- 정규화는 최소값과 최대값을 이용하다보니 이상치에 민감하다. 따라서 정규화 전에 이상치를 제거해야 한다.
- 표준화는 상대적으로 이상치에 둔감하다. 따라서 표준화 이후에 이상치를 제거하면 된다.
- 만약 정규화와 표준화를 동시에 적용한다면, 표준화 이후 이상치를 제거한 뒤, 정규화를 진행한다.


# 4. 결론

- ML/DL 하면서는, scaling은 필수라고 생각해도 괜찮을 정도인 듯. 다만 덮어놓고 바로 정규화부터 적용하기보다는, 모델 가정과 데이터 특성들 살펴보면서 왜 적용하는지 매번 확인하도록 하자.
- 위에서 설명한 min-max normalization, Z-score standardization 외에도 다양한 scaling 방법 있으니, 상황에 가장 적합한 방법이 뭘지 항상 고민해보자.
- 언제 어떤 scaling 방법을 적용하면 되는지 구체적인 규칙을 알고 싶었지만, case-by-case라는 답이 주류인 걸로 보인다. 표준화, 정규화에 대한 정해진 Rule-book은 없다. 다양한 케이스 접하면서, 실제로 이들 방법에 따른 성능 차이가 어떻게 나타나는지 확인하면서 경험을 쌓아보자.

# Reference

- [데이터의 정규화(normalization) 또는 표준화(standardization)이 필요한 이유](https://mozenworld.tistory.com/entry/%EB%8D%B0%EC%9D%B4%ED%84%B0%EC%9D%98-%EC%A0%95%EA%B7%9C%ED%99%94normalization-%EB%98%90%EB%8A%94-%ED%91%9C%EC%A4%80%ED%99%94standardization%EC%9D%B4-%ED%95%84%EC%9A%94%ED%95%9C-%EC%9D%B4%EC%9C%A0)
- [정규화 정리1 - Scaling, Regularization, Standardization | 너드팩토리 블로그](https://blog.nerdfactory.ai/2021/06/15/Normalization-Theorem-1.html)
- [machine learning - Zero Mean and Unit Variance - Data Science Stack Exchange](https://datascience.stackexchange.com/questions/32109/zero-mean-and-unit-variance)
- [machine learning - Why do we have to normalize the input for an artificial neural network? - Stack Overflow](https://stackoverflow.com/questions/4674623/why-do-we-have-to-normalize-the-input-for-an-artificial-neural-network/4674770#4674770)
- [comp.ai.neural-nets FAQ, Part 2 of 7: LearningSection - Should I normalize/standardize/rescale the](http://www.faqs.org/faqs/ai-faq/neural-nets/part2/section-16.html)
- [What is Feature Scaling and Why is it Important](https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-standardization/)
- [About Feature Scaling and Normalization](https://sebastianraschka.com/Articles/2014_about_feature_scaling.html#about-standardization)
- [정규화, 표준화 질문드립니다. - 인프런 | 커뮤니티 질문&답변](https://www.inflearn.com/community/questions/65387/%EC%A0%95%EA%B7%9C%ED%99%94-%ED%91%9C%EC%A4%80%ED%99%94-%EC%A7%88%EB%AC%B8%EB%93%9C%EB%A6%BD%EB%8B%88%EB%8B%A4)
- [데이터 전처리의 피처 스케일링(Feature Scaling)](https://glanceyes.com/entry/%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%A0%84%EC%B2%98%EB%A6%AC%EC%9D%98-%ED%94%BC%EC%B2%98-%EC%8A%A4%EC%BC%80%EC%9D%BC%EB%A7%81Feature-Scaling)
- [Machine Learning: When should I apply data normalization/standardization? - Quora](https://www.quora.com/Machine-Learning-When-should-I-apply-data-normalization-standardization)
- [When do you normalize and when do you standardize the features of a dataset? - Quora](https://www.quora.com/When-do-you-normalize-and-when-do-you-standardize-the-features-of-a-dataset)