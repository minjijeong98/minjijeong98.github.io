---
title: Seq2Seq with Attention 모델 구현 Flow 별 주요 내용 정리
description: 부스트캠프의 과제로 seq2seq with attention 모델을 구현하면서, 각 flow 별 주요 내용 기록
author: mj
date: 2024-09-06 01:50:00 +0900
categories: [AI, NLP]
tags: [implementation, nlp, seq2seq, attention]
use_math: true
pin: true
---
> 부스트캠프의 과제로 **seq2seq with attention**를 구현했습니다. 해당 과제를 진행하며 전체적인 흐름을 이해하는데 가장 중점을 두었습니다. 모델 알고리즘을 코드로 어떻게 구현하는지, 구현한 모델을 어떻게 학습/평가/시각화하는지를 확인했고, 각 과정에서 기억해둘 내용을 기록합니다.

## 개요
### 핵심 모델 구조
- **encoder**: bidirectional GRU
- **decoder**: unidirectional GRU
- **attention**: dot-product attention

### 코드 구성 흐름
0. 데이터 로드
1. 데이터 전처리
	1. 데이터 토큰화
	2. 어휘 사전 구축
	3. 어휘 사전을 활용한 데이터 토큰 ID 변환
2. Seq2Seq with Attention 구현
	1. Encoder 구현
	2. Dot-product attention 구현
	3. Decoder 구현
	4. Seq2Seq 모델 구축
3. 모델 학습
	1. 모델 weight 초기화
	2. Data loader 구현
	3. Train 함수 구현
	4. evaluate 함수 구현
	5. 전체 모델 학습, 평가, 시각화 (attention map)


## 0. 데이터 로드
- `datasets` 라이브러리 > `bentrevett/multi30k` 데이터셋 사용
	- 영어, 독일어로 구성된 번역 데이터셋
	- 기계번역 및 seq2seq 모델 학습에 주로 사용됨

## 1. 데이터 전처리
- 신경망이 문자열을 처리할 수 있도록 숫자로 변환하는 과정
	- 문자열을 토큰으로 변환한 다음,
	- 이들 토큰을 모아 어휘 사전을 만들고
	- 이 어휘 사전을 look up table로 사용하여 각 시퀀스의 토큰들을 숫자로 변환
- `spacy` 라이브러리의 영어, 독일어 토큰화기 활용
	- 자연어 처리를 위한 오픈소스 라이브러리
	- 효율적이고 빠른 텍스트 처리 기능 제공
	- 토큰화, 형태소 분석, 품사 태깅, 의존 구문 분석 등 지원
- train, validation, test 각각에 대해 데이터 전처리 수행

### 1-1. 데이터 토큰화
1. 각 시퀀스에 대해 토큰화 수행하는 함수 정의
	1. `spacy`의 `tokenizer()` 사용해 각 시퀀스 대해 토큰화 수행
	2. `lower = True` 면 각 토큰 소문자로 변환 (`lower()`)
	3. 토큰 앞뒤로 `sos_token`, `eos_token` 추가
2. 데이터셋 전체에 대해 1의 함수 적용하는 함수 정의

### 1-2. 어휘 사전 구축
- source 및 target language에 대한 어휘 사전 구축
	- 데이터셋의 각 unique token을 index(정수)로 연결시키는 데 사용됨
	- 예: "hello" = 1, "world" = 2, "bye" = 3, "hates" = 4 등
- `torchtext` 에서 제공하는 `build_vocab_from_iterator()` 함수 사용
- 어휘 사전은 train set으로만 구축해야 함 (valid, test 안 됨)
	- valid, test set도 이용하면 information leakage 발생해서 모델 성능 부풀려질 수 있음에 유의
	- 만약 학습셋에 없는 토큰이 valid, test 데이터셋에서 등장했다면, `<unk>` 토큰으로 대체 (일반적으로 인덱스 0)
	- 모델이 `<unk>` 토큰에 대해 주변 문맥을 사용해 번역하는 방법을 학습시키기 위해, 학습셋에도 `<unk>` 토큰을 포함시킴
	- 이를 위해 `build_vocab_from_iterator()`로 어휘사전 구축 시 `min_freq` 인수를 사용해, trainset에서 `min_freq` 횟수보다 적게 나타나는 토큰은 `<unk>`로 처리
- `build_vocab_from_iterator()`의 `specials` 인수로 special token 전달
	- 어휘 사전에 추가하고 싶지만 토큰화된 예제에 반드시 나타나지는 않을 수 있는 토큰
	- `<pad>`: padding token
		- 모델에 문장 입력 시, 여러 문장을 한 번에 전달("batch")하는 것이 효율적임
		- 다만 배치 처리를 위해서는 문장의 길이가 모두 같아야 한다는 조건 발생 (토큰 수 기준)
		- 대부분의 문장 길이는 서로 다르므로, 패딩 토큰 사용해 이런 조건 처리
		- 배치에서 가장 긴 문장과 동일한 길이 갖도록, 나머지 문장들에 패딩 토큰 추가
- `build_vocab_from_iterator()`로 생성된 vocab 세부 내용 확인
	- `get_itos()`: int to string. 토큰 확인
	- `get_stoi()`: string to int. 인덱스 확인
	- `len()`: 어휘 사전의 unique token 수 확인
	- 어휘사전의 앞부분은 special token
	- 그 이후는 가장 많이 등장한 순서부터 내림차순 정렬

```python
en_vocab = torchtext.vocab.build_vocab_from_iterator(
    train_data["en_tokens"],
    min_freq=min_freq,
    specials=special_tokens,
)
en_vocab.get_itos()[:10]  # ['<unk>', '<pad>', '<sos>', '<eos>', 'a', '.', 'in', 'the', 'on', 'man']
en_vocab.get_stoi()["the"]  # 7
len(en_vocab), len(de_vocab)  # (5893, 7853)
```

### 1-3. 어휘 사전을 활용한 데이터 토큰 ID 변환
- **"numericalize"**: 토큰을 인덱스로 변환하는 방법
	- `lookup_indices`: 토큰 리스트 -> 인덱스 리스트
	- `lookup_tokens`: 인덱스 리스트 -> 토큰 리스트
- 한편, feature를 올바른 데이터 타입으로 변환해줘야 함
	- numericalize된 인덱스는 파이썬 기본 정수형임
	- pytorch에서 사용하기 위해 `with_format()` 사용해 pytorch 텐서로 변환


## 2. Seq2Seq with Attention 구현
### 2-1. Encoder 구현
1. 배치 임베딩 & 드롭아웃
2. $h_0$ 초기화
3. GRU 통해 output, hidden 구하기
4. 3의 결과값 적절히 처리해 필요한 최종 정보 return
	- **output**: 각 시퀀스의 모든 time-step에 대한 인코딩 정보 포함
	- **hidden**: 전체 시퀀스의 최종 인코딩 정보 통합

### 2-2. Dot-product attention 구현
1. $H$의 각 행과 $s_t$의 내적으로 에너지 계산
	- $s_t$: 현재 time-step의 hidden state
	- $H$: 인코더의 모든 전방 & 후방 hidden state
2. 1의 결과에 softmax 적용해 attention score 벡터 $a_t$ 생성
3. attention value $\hat{a_t}$ 계산: $H$ 의 각 행에 가중치 곱해서 더하기

### 2-3. Decoder 구현
1. 디코더의 현재 hidden state $s_t$ 연산
2. 현재 디코더 hidden state의 attention value 연산
3. 1과 2의 결과 concat해 최종 output 예측 (Linear)
4. 디코더 output, hidden, attn_scores 리턴

### 2-4. Seq2Seq 모델 구축
1. 인코더 적용 -> 인코더 output, hidden 얻기
2. 디코더 적용 -> 디코더 output, hidden, attn_scores 얻기
	- 이 때, 디코더에는 attention 반영해서 최종 결과 내도록 이미 2-3에서 처리된 상태
3. 디코더의 각 time-step에서 최대 확률값 갖는 토큰 id 반환
- 이 때, teacher forcing 이용해 추론한 값 뿐만 아니라 ground truth도 사용될 수 있도록 처리함

### Teacher Forcing
- seq2seq 모델 학습에서 exposure problem을 처리하기 위해 자주 사용되는 기술
- **exposure problem**
	- 디코더가 학습 중에 target sequence 초기에 부정확한 단어 생성할 수 있기 때문에 발생
	- 이런 부정확한 단어는 이후 단계에 대한 디코더의 입력이 되어 모델을 잘못된 경로로 이끌 수 있음
- **teacher forcing**
	- seq2seq 모델 학습 시 decoder가 다음 state를 예측할 때, 이전 time step의 예측값이 아닌 실제 target sequence (ground truth)를 입력으로 사용하는 기법
	- 학습 시에는 teacher forcing 사용하지만, 평가 시에는 사용하지 않음
	- 학습 중 teacher forcing 비율을 조정하여 점진적으로 모델이 실제 사용 조건에 적응하도록 할 수 있음
	- **장점**: 학습을 빠르고 안정적으로 만듦
	- **단점**: 학습과 추론 단계의 차이로 인해 모델 성능과 안정성 낮아질 수 있음


## 3. 모델 학습
### 3-1. 모델 weight 초기화
- (거의) 영벡터로 초기화

### 3-2. Data loader 구현
- padding 반영해서 data loader 구현
	- `nn.utils.rnn.pad_sequence` 사용
- batch size 지정
- train, valid, test set 로드
- train set에서 source(출발어; 영어)와 target(도착어; 독일어) 각각의 최대 시퀀스 길이 확인

### 3-3. Train 함수 구현
- train에서는 teacher forcing 적절히 사용
- cross entropy loss 활용하되, Padding token은 학습 Loss에 관여하지 않도록 처리

```python
criterion = nn.CrossEntropyLoss(ignore_index = corpus.pad_index)
```
### 3-4. evaluate 함수 구현
- teacher forcing 사용하면 안됨

### 3-5. 전체 모델 학습 진행


## 느낀점
#### 코드 구현까지 할 수 있어야 제대로 이해한 것
- 이론으로만 이해했을 때와, 이를 코드로 구현하는 건 다른 느낌이었다.
- 이론으로 큰 그림을 그렸다면, 코드 통해 디테일을 잡아가는 느낌
- 지금까지는, _이론->코드_ 순서로 모델을 보는게 더 잘 이해되는 것 같다.

#### 차원 맞추기가 어렵다
- 처음 과제 풀 때, 생각하는 논리 구조대로 우선 무작정 코드를 작성했더니 계속 차원이 달라서 계산이 불가능하다는 에러가 발생했다.
- 그냥 돌아가게 하는 목적으로 squeeze와 unsqueeze 활용해 겉모양은 맞출 수 있겠지만, '제대로' 하려면 모델의 구조 제대로 이해해야 하고, 사용하는 코드들의 output 형상도 잘 이해해야겠다고 생각했다
- 과제였던 만큼, 각 과정에서 필요한 변수의 크기가 주석으로 세세하게 달려있어서 많은 도움을 받았다. 이 주석에 의지하지 않고도 차원 맞출 수 있을 정도로 잘 이해해보자!

#### pytorch에서 자주 사용되는 메소드의 구성을 살펴보고 싶어졌다
- `nn.Embedding`, `nn.Linear` 등, 자주 사용되는 메소드들에 대해서는 어떤 구조로 작동하는지 확인해보고싶어졌다. '다들 그렇게 하니까..'라는 식으로 넘기고 싶지 않음

#### attention의 영향?
- attention을 적용한 버전의 seq2seq를 구현했는데, attention이 없는 버전에 비해 얼마나 성능이 좋아진건지, 수치화해보고 싶어졌다. (수치화가 힘들다면 정성평가라도..) 
- 기본 seq2seq 모델을 코드로 짜고, 성능 비교할 수 있는 적절한 metric 선정해서 성능 비교해보자!


## Reference
- 부스트코스 강의자료 및 과제
- [Sequence to Sequence Learning with Neural Networks.ipynb](https://github.com/bentrevett/pytorch-seq2seq/blob/main/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb)
- [Sequence-to-Sequence Models for Language Translation](https://www.analyticsvidhya.com/blog/2024/05/sequence-to-sequence-models-for-language-translation/)