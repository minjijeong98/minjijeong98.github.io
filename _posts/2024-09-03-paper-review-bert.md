---
title: (논문리뷰) BERT; Pre-training of Deep Bidirectional Transformers for Language Understanding
description: Transformer의 주요 활용 사례로 언급되는 BERT 모델 리뷰
author: mj
date: 2024-09-03 23:45:00 +0900
categories: [AI, NLP]
tags: [paperreview, nlp, bert]
use_math: true
pin: true
---

# 논문 개요
- 제목: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
- 기관: Google AI Language
- 발표: 2018.10
- 인용수: 110,833회 (2024.09.03 기준)
- 코드: [google-research/bert](https://github.com/google-research/bert)

# 핵심 내용 요약
- BERT (Bidirectional Encoder Representations from Transformers) 모델 제안
	1. **pre-training**: unlabeled text data에서 deep bidirectional representation 학습
	2. **fine-tuning**: 하나의 output layer만을 추가해, 특정 downstream task에 대해 labeled data 이용해 transfer learning
- 결과: 11개의 주요 NLP task에서 SoTA 달성 (문장수준, 토큰수준 Task 모두)
- 연구의의: task-specific architecture보다 저렴하면서도 뛰어난 성능 보이는 첫번째 fine-tuning 기반 representation 모델

# 주요 용어 정리
- **self-supervised learning**: 데이터의 일부를 숨기고, 해당 부분을 예측하도록 학습
- **pre-training**: self-supervised learning 기법을 이용하여 사전에 모델 학습. 데이터로부터 유의미한 지식 배웠을 것이라는 가정 내포.
- **transfer learning**: 사전학습된 모델을 downstream task에 대해 fine-tuning하는 방법론
- **feature-based**: 임베딩은 그대로 두고, 그 위의 layer만 학습하는 방법. task-specific architecture를 구성하고, 거기에 pre-trained representation (즉, embedding layer)를 부가적인 feature로 사용하는 방법
- **fine-tuning**: 모두 (그러나 미세하게) 업데이트하는 방법. 임베딩까지 모두 업데이트 함
- **objective function**: 

# 모델 등장 배경
### Pre-Training 방법의 우수한 성능
- BERT 이전에도 Language model (LM)의 pre-training은 다수 NLP task에서 우수한 성능 보여왔음
	- 문장 수준 task: 문장의 관계 전체적으로 분석해 예측하는 것을 목표로 하는 task (예: NLI, paraphrasing)
	- 토큰 수준 task: 토큰 수준의 세분화된 출력을 생성하는 task (예: NER, QA)
- downstream task에 pre-trained language representation 적용하는 두가지 접근
	1. **feature-based approach**
		- 주로 unlabeled text로 부터 word embedding parameter를 pre-training하는 방식
		- 예시: [ELMo](https://aclweb.org/anthology/N18-1202) (Embeddings from Language Model)
			- cue: Word2Vec 방식의 임베딩 벡터는, 문맥에 따른 단어의 의미 차이 반영하지 못한다. 예를 들어 Bank Account(은행 계좌)와 River Bank(강둑)에서의 Bank는 서로 다른 의미를 갖는다.
			- idea: "문맥을 반영해 word embedding 한다"
				- 각 층의 출력값이 가진 정보는 전부 서로 다른 종류의 정보를 갖고 있을 것이므로, 이들을 모두 활용한다
			- biLM: 순방향, 역방향 언어모델을 별개의 2개 언어모델로 보고 학습 
				- ![elmo-1](/assets/img/elmo-1.png)
			- 순방향 언어모델과 역방향 언어모델 각각에서 해당 time step의 biLM의 각 층의 출력값을 가져와 concat하고, 가중합 => "ELMo representation"
				- ![elmo-2](/assets/img/elmo-2.png)
			- BiLM 통해 얻은 ELMo representation과 기존 임베딩 벡터를 concat해서 input으로 사용 (shallow bidirectional)
				- ![elmo-3](/assets/img/elmo-3.png)
				- left-to-right, right-to-left language model을 단순히 concat한 ELMo representation ![elmo-4](/assets/img/elmo-4.png)
			- task-specific architecture: downstream task에서 pre-trained representation을 하나의 추가적인 feature로 활용
	2. **fine-tuning approach**
		- contextual token representation을 만들어내는 (문장 혹은 문서)인코더가 pre-training 되고, supervised downstream task에 맞춰 fine-tuning 되는 방식
		- 장점: scratch로 (처음부터) 학습하는데 비해 적은 파라미터로도 충분
		- 예시: OpenAI GPT
		- task-specific (fine-tuned) parameter 수는 최소화하고, 모든 pre-trained parameter를 조금만 바꿔서 downstream task를 학습
	- feature based approach와 fine-tuning approach 모두 pre-train 과정에서 동일한 목적 함수 (objective function)을 공유하는데, 이 때 general language representation 학습하기 위해 unidirectional language model 사용함

### 기존 연구의 한계
- BERT, GPT, ELMo 비교 ![bert-comparison](/assets/img/bert-comparison.png)
- 기존 방식은 unidirectional 이라는 점에서 pre-training 중에 사용할 수 있는 architecture 구조가 제한되어 pre-trained representation의 성능 제한
- 특히 GPT의 경우, 모든 토큰이 이전 토큰들과의 attention으로만 계산하므로(**constrained self-attention**) 토큰이 아닌 문장 수준의 정보를 반영해야 하는 다음 문장 예측 task나, 양방향의 context 통합이 중요한 QA task와 같은 토큰 수준 작업에서 성능 한계 보임
- 논문에서는 특히 fine-tuning 접근방식에 집중, 기존의 unidirectional 문제 개선해 deep bidirectional context를 반영하는 BERT 제안 (**bidirectional self-attention**)
> [!important] "deep" bidirectional
> ELMo가 순방향 언어 모델과 역방향 언어 모델을 모두 사용하기 때문에 Bidirectional lanuage model이라고 생각할 수 있지만, ELMo는 각각의 단방향(순방향,역방향) 언어모델의 출력값을 concat해서 사용하기 때문에 하나의 모델 자체는 단방향이다. 이것이 바로 BERT에서 강조하는 deep bidirectional과의 차이점이라고 할 수 있다.

> [!important] GPT vs. BERT
> bidirectional Transformer는 보통 "Transformer encode"라고 불리고, left-context-only version은 "Transformer decoder"라고 불림 (text 생성에 사용될 수 있으므로)
> 참고로 **GPT**는 next token만을 맞추는 기본적인 language model을 만들기 위해 **transformer decoder**만을 사용하였고, **BERT**는 MLM과 NSP를 위해 self-attention을 수행하는 **transformer encoder**만을 사용했다.

# 4. 모델 구조
![bert-1](/assets/img/bert-1.png)

### Multi-layer bidirectional Transformer encoder
- 양방향 Transformer encoder를 여러 층 쌓음
- 다양한 Task에 대해 통합된 architecture
	- Output layer를 제외하고는 pre-training과 fine-tuning 모두에 동일한 아키텍처가 사용됨
	- 서로 다른 downstream task에 대해, 항상 동일한 pre-trained model의 파라미터가 초기값으로 사용되고, fine-tuning 과정에서 이 파라미터들이 downstream task에 맞게 조정됨
- 특수 토큰
	- `[CLS]`: 모든 input example 앞에 추가되는 특수 토큰
	- `[SEP]`: 특수 구분 토큰 (예: 질문/답변 구분)
- notation
	- $L$: layer(즉, transformer block)의 수
	- $H$: hidden size
	- $A$: self-attention head의 수
	- $E$: input embedding
	- $C \in \mathbb{R}^H$: 특수 토큰 `[CLS]`의 최종 hidden vector
	- $T_i \in \mathbb{R}^H$: $i$번째 input token에 대한 최종 hidden vector
- 본 논문에서는 2가지 모델 크기 고려
	- $\text{BERT}_{\text{BASE}}$: $L=12$, $H=768$, $A=12$, 전체 파라미터 110M개
	- $\text{BERT}_{\text{LARGE}}$: $L=24$, $H=1024$, $A=16$, 전체 파라미터 340M개

### Step 0. Input/Output Representation
![bert-2](/assets/img/bert-2.png)
- input sequence는 하나의 문장일 수도, 두 문장의 쌍일 수도 있음 (QA 등의 다양한 downstream task 처리하기 위해 한가지 형태로 제한하지 않음)
	- 문장 쌍의 각 문장은 `[SEP]` 토큰으로 구분됨
- 3가지의 embedding vector 합쳐서 input representation으로 사용 (token/segmentation/position embedding)
	- 모든 시퀀스의 첫번째 토큰은 항상 `[CLS]` 토큰 (NSP를 위한 토큰임)
	- token embedding: WordPiece 임베딩 (3만 개의 토큰 어휘 사용)
	- segmentation embedding: 각 토큰이 문장 A에서 나왔는지, B에서 나왔는지 구분하기 위함
	- position embedding: 단어의 위치 구분하는 임베딩

### Step 1. Pre-Training BERT
- Masked LM(MLM)과 Next Sentence Prediction(NSP)의 두 가지 unsupervised-task 동시에 사용해 pre-training 수행
- 해당 task들 통해 단방향성 제약이 완화됨 (Ablation study 통해 MLM과 NSP 모두에 대한 유의미한 성능 향상 확인함)

#### Task 1: Masked LM (MLM)
![bert-3](/assets/img/bert-3.png)
- 아무런 제약 조건 없이 bidirectional하게 학습 하면, 간접적으로 예측 대상 단어를 참조하게 되어 학습이 어려움. 이에 penalty 부여하는 방법으로써 MLM task 도입함.
- MLM: input token의 일정 비율을 랜덤하게 마스킹하고, context 통해 이 마스킹된 토큰을 예측하는 task
	- Cloze Test (빈칸 채우기 테스트)에서 영감 얻음
	- 예: `오늘, 나는 ___에 가서 우유와 계란을 샀다. 비가 올 줄 알았지만 ___을 챙기는걸 잊어서 가는 길에 젖었다 (들어갈 단어: 마트, 우산)`
	- 마스킹된 구절에 속하는 올바른 언어/품사 식별 위해서는 context와 어휘 이해하는 능력이 필요
- 각 입력 시퀀스의 15% 토큰을 무작위로 마스킹 (15%라는 수치는 경험적으로 결정)
	- BERT는 오직 `[MASK]` 토큰만을 예측
	- `[MASK]` 토큰은 Pre-training에서만 사용됨 -> pre-training과 fine-tuning 사이의 불일치 완화하기 위해 일부는 `[MASK]`가 아닌 다른 토큰으로 마스킹함
		- 80%: `[MASK]` 토큰 (예: my dog is *hairy* -> my dog is *[MASK]*)
		- 10%: 랜덤 토큰 (예: my dog is *hairy* -> my dog is *apple*)
		- 10%: 원래 토큰 (예: my dog is *hairy* -> my dog is *hairy*)
			- 실제 관측 단어에 대한 representation을 bias해주기 위함. `[MASK]`와 랜덤 토큰만 사용하면, 모델이 '무조건 새로운 단어로 바꾼다'고 학습할 수 있으므로.
- 마스킹된 토큰에 대응되는 최종 hidden vector $T_i$를 이용해 cross entropy loss 기반으로 원래 토큰 예측

#### Task 2: (binarized) Next Sentence Prediction (NSP)
![bert-1](/assets/img/bert-1.png)
- 많은 NLP downstream task(QA, NLI 등)의 핵심은, 두 문장 간의 관계를 이해하는 것
- 이를 위해 문장 A의 뒤에 문장 B가 실제 이어지는 문장인지 예측함
- 연속된 문장 쌍과, 그렇지 않은 문장 쌍(corpus에서 랜덤하게 선택)을 1:1 비율로 입력해 학습시킴
	- 연속된 문장쌍 input sequence (label: `IsNext`)
		- `[CLS] the man went to [MASK] store [SEP] he bought a gallon [MASK] milk [SEP]`
	- 불연속 문장쌍 input sequence (label: `NotNext`)
		- `[CLS] the man [MASK] to the store [SEP] penguin [MASK] are flight ##less birds [SEP]`
- $C$는 특수 토큰 `[CLS]`의 최종 hidden vector로써, input으로 들어온 두 문장이 원래 연속된 문장인지(`IsNext`) 아닌지(`NotNext`)를 맞춰가며 학습
- 논문의 Ablation Study에서는 해당 NSP 학습이 없으면 모델 성능 감소한다고 명시함

#### Pre-training data
- pre-training 과정은 많은 데이터를 필요로 함
- corpus를 구축하기 위해 BookCorpus(800M)와 English Wikipedia (2,500M)를 사용함
- 대량의 unlabeled data로 학습을 진행 (self-supervised learning)
- 기존 pre-trained LM의 접근방식 따름

### Step 2. Fine-Tuning
- parameter를 pre-trained parameter로 초기화한 뒤, downstream task에 대한 labeled data 이용해 parameter를 fine-tuning
- downstream task별 input, output
	- entailment: 가정-전제 쌍의 함의 관계에 대한 분류 task (내포/모순/무관 중 하나로 분류)
	- tagging: 문장의 각 단어에 대한 품사 태깅, entity 태깅 (NER) 등
	- text classification: 문장 간의 관계를 예측

| downstream task                            | input (pairs)      | output               |
| ------------------------------------------ | ------------------ | -------------------- |
| paraphrasing                               | sentence           |                      |
| entailment<br>(classification)             | hypothesis-premise | [CLS] representation |
| QA                                         | question-passage   | token representation |
| text classification / <br>sequence tagging | degenerate-none    | token representation |

- fine-tuning은 pre-training에 비해 훨씬 저렴함 (대부분의 task에서 TPU 1시간, GPU 몇시간 이내 소요)

- 텍스트 쌍과 관련된 응용의 경우 일반적인 패턴은, bidirectional cross attention을 적용하기 전에 텍스트 쌍을 독립적으로 인코딩하는 것
- 하지만 BERT에서는 self-attention 매커니즘을 사용하여 이 두 단계를 통합함
	- self-attention으로 concat된 텍스트 쌍을 인코딩하면 두 문장 간의 bidirectional cross attention이 효과적으로 포함되기 떄문

# 5. 실험 결과
### GLUE (General Language Understanding Evaluation) 
![bert-result-1](/assets/img/bert-result-1.png)
- 다양한 자연어 이해 task의 모음
	- 참고: [GLUE leaderboard](https://gluebenchmark.com/leaderboard) ~~2024.09 현재 시점에서는BERT가 49위~~
- $\text{BERT}_{\text{BASE}}$와 $\text{BERT}_{\text{LARGE}}$ 모두 기존 방법보다 우수한 성능 보임
	- $\text{BERT}_{\text{BASE}}$와 OpenAI GPT는 attention masking을 제외하고 모델 아키텍처 측면에서 거의 동일하다는 점에서, 흥미로움

### SQuAD(Stanford Question Answering Dataset)
![bert-result-2](/assets/img/bert-result-2.png)
![bert-result-3](/assets/img/bert-result-3.png)
- (question, passage) 데이터 주어지면 passage 내에서 answer span을 예측하는 task
	- 정답의 시작과 끝 토큰 예측하는 classifier 사용
- v1.1: passage에 답변 항상 존재
- v2.0: v1.1을 확장해서, passage에 답변 없을 가능성 허용
	- 정답이 없는 문제의 answer span: `[CLS]`~`[CLS]`
	- 인간만큼은 아니지만, 기존 baseline에 비해서는 우수한 성능 확인

### SWAG (Situations With Adversarial Generations)
![bert-result-4](/assets/img/bert-result-4.png)
- 상식 추론: 문장이 주어지면, 4개의 선택지 중 가장 그럴듯하게 이어지는 문장 고르는 task


### Ablation Studies
- BERT의 facet의 상대적 중요성 이해하기 위해 3가지 제거 실험 수행함
	- pre-training task의 영향 (MLM, NSP): 중요함
	- 모델 크기의 영향: 모델 클수록 성능 좋음
	- BERT의 feature 기반 접근방식: fine-tuning과 성능 유사함
- BERT의 feature 기반 접근방식 ![bert-result-5](/assets/img/bert-result-5.png)
	- BERT는 fine-tuning 및 feature-based approach 모두에서 효과적임


# Review


# Reference
- [엘모(Embeddings from Language Model, ELMo)](https://wikidocs.net/33930)
- [Cloze test](https://en.wikipedia.org/wiki/Cloze_test)
- [\[최대한 자세하게 설명한 논문리뷰\] BERT (1)](https://hyunsooworld.tistory.com/entry/%EC%B5%9C%EB%8C%80%ED%95%9C-%EC%9E%90%EC%84%B8%ED%95%98%EA%B2%8C-%EC%84%A4%EB%AA%85%ED%95%9C-%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0-BERT-Pre-training-of-Deep-Bidirectional-Transformers-for-Language-Understanding-1)
- [\[NLP | 논문리뷰\] BERT 상편](https://velog.io/@xuio/NLP-%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0-BERT-Pre-training-of-Deep-Bidirectional-Transformers-forLanguage-Understanding-%EC%83%81%ED%8E%B8), [하편](https://velog.io/@xuio/NLP-%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0-BERT-Pre-training-of-Deep-Bidirectional-Transformers-for-Language-Understanding-%ED%95%98%ED%8E%B8)