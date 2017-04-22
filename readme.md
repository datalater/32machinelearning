copyright ⓒ JMC 2017

규칙1 : 완벽한 노트를 작성하겠다는 마인드는 버린다.  
규칙2 : 말로 설명할 수 있도록 스토리텔링 형식으로 적는다.  
규칙3 : 오직 지금의 나를 위한 노트를 작성한다.  

주요 소스 : Coursera ML by Andrew Ng, ML 강의노트 by 박수진  
참고 소스 : 테리의 딥러닝 토크, 카카오 AI 리포트

---

# 2강 단순회귀

## 용어 정리

+ 선형회귀 : lnear regression
    + supervised learning에 속하고
    + output이 연속적인 값이면서
    + input과 output 사이에 선형관계가 존재할 때 사용한다.


+ 단순회귀 : univariate linear regression
    + 선형회귀 조건을 만족하면서
    + input 변수가 1개일 때 사용한다.
    + 실제로는 거의 사용되지 않고, 이론적인 이해를 돕기 위해 배운다.

    > 더 쉽게 linear regression with one variable이라고 표현해도 된다.


+ 다중회귀 : multivariate linear regression
    + 선형회귀 조건을 만족하면서
    + input 변수가 여러 개일 때 사용한다.
    + 실제로 굉장히 많이 사용되는 학습방법이다.

## 선형회귀는 언제 사용하는가?
+ supervised learning
    + continuous ouput
        + input과 output 사이에 선형관계가 존재한다고 가정할 때

## 모델과 데이터

+ 스토리텔링

```
모델이란 무엇인가?
주어진 데이터를 표현한 것을 뜻한다.

모델이 데이터를 잘 표현하는지 눈으로 확인하기 위해서 그래프를 사용한다.
그래프에 찍히는 (x, y) 점은 주어진 데이터를 뜻하고
그래프에 표현되는 직선 또는 곡선은 모델을 뜻한다.

* 주어진 데이터 = training data set = training set
* x = input variable
* y = output variable = target variable
```

## Model Representation

+ model representation : 모델을 만드는 방법


+ 스토리텔링

```
좋은 단순회귀 모델이란 무엇인가?
주어진 데이터를 잘 표현하는 직선을 뜻한다.

모델을 만드는 방법은 무엇일까?
직선의 방정식을 구하면 된다.

* 직선의 방정식 = function = hypothesis = hypothesis function = h = h(x)


직선의 방정식을 구하려면 절편 값과 기울기를 알아야 한다.
여기서 기울기를 parameter라고 한다.
parameter에 어떤 값이 들어가느냐에 따라 선형회귀 모델이 달라진다.

parameter를 알맞게 정하는 방법은 무엇일까?
주어진 데이터와 직선의 오차를 구하는 방정식인 cost function을 이용하면 된다.

cost function 중 가장 많이 쓰이는 방법은 LSM이다.
LSM은 Least Squared Method로써, 오차를 제곱한 값을 합했을 때 가장 적은 수치를 찾는 방법을 뜻한다.

LSM이 왜 가장 많이 쓰일까?
오차를 제곱하지 않으면 양수와 음수가 존재할 수 있는데,
양수와 음수를 더하면 실제 오차의 값이 상쇄되어 버린다.
이러한 이슈를 방지하는 가장 쉬운 방법으로써 LSM이 쓰인다.

LSM을 사용해서 구한 오차를 LSE라고 한다.
LSE는 Least Squared Error를 뜻한다.

```

@@@ resume : https://www.coursera.org/learn/machine-learning/lecture/rkTp3/cost-function



---

# 1강 Introduction

## 머신러닝이란 무엇인가?

+ 프로그래머가 직접 수많은 규칙을 미리 정해주는 대신 프로그램 자체가 데이터를 통해 스스로 학습하도록 하는 방법

## 데이터셋이란 무엇인가?

+ 스토리텔링

```
하나의 데이터는 n개의 변수를 가지고 있다.
이러한 데이터가 모여있는 것을 데이터셋이라고 한다.
즉, 데이터셋은 n개의 변수를 갖고 있는 데이터들의 집합이다.

이때 데이터의 변수가 input과 output으로 나눠지는 경우도 있고, 나눠지지 않는 경우도 있다.
만약 변수 A가 변수 B를 예측하는 데 효과가 있다면, 변수 A는 input이 되고 변수 B는 output이 된다.
이때 output 변수는 예측의 목적이 되므로 target 변수라고도 한다.

예를 들어, 종양을 관찰하면 하나의 데이터가 완성된다.
종양의 크기, 종양의 색깔, 종양의 부피 등이 종양 데이터의 변수가 된다.
악성(malignant) 또는 양성(benign)도 종양 데이터의 변수가 된다.
크기, 색깔, 부피는 종양 데이터의 input 변수가 되고
악성 또는 양성은 데이터의 output 변수가 된다.
종양 데이터를 여러 개 모아두면 종양 데이터셋이 만들어진다.

관찰한 데이터를 sample 데이터라고 한다.
sample 데이터를 잘 학습해서 실제 현실을 예측하는 것이 학습의 목표이다.
```

## Supervised Learning

+ 특징 : 데이터셋의 output에 label이 있는 경우


+ supervised learning의 세부 분류
    + regression
    + classification


+ regression vs. classification
    + regression : continuous output
    + classification : discrete output


+ 스토리텔링

```
supervised learning은 데이터셋의 변수가 input과 output으로 나눠져야 사용할 수 있다.
supervised learning은 input으로 output을 예측하기 위해 학습한다.
supervised learning은 input이 output에 미치는 영향을 계량화한다.

이때 output이 연속적인 값이라면 regression을 사용하고,
output이 분류되어야 하는 범주 값이라면 classification을 사용한다.

예를 들어, 집에 대한 데이터가 있다.
집의 크기, 위치, 건축연도 등이 input 변수이고
집값이 output 변수가 된다.
크기, 위치, 건축연도 등으로 집값을 예측하려면
집값이 연속적인 값이므로 regression을 사용하면 된다.
```

## Unsupervised Learning

+ 특징 : 데이터셋의 output에 label이 없는 경우


+ unsupervised learning의 세부 분류
    + clustering
    + non-clustering


+ 스토리텔링

```
unsupervised learning은 데이터셋의 변수가 input과 output으로 나눠지지 않을 때 사용할 수 있다.
unsupervised learning은 변수를 통해 데이터 간의 관계를 파악하기 위해 학습한다.
unsupervised learning은 데이터의 변수들이 서로 유사(similar)하거나 관계(related)가 있는 정도를 계량화한다.

이때, 섞여 있는 데이터셋에서 서로 비슷한 데이터를 그룹화하는 것을 clustering이라 한다.
섞여 있는 데이터셋에서 특정 데이터를 찾아내는 것을 non-clustering이라 한다.

예를 들어, 신문사들이 쓴 기사를 데이터셋으로 학습한다고 해보자.
기사를 관찰하면 신문사, 기자 이름, 날짜, 제목, 기사 내용으로 변수를 구성할 수 있다.
이러한 변수들은 input과 ouput의 관계를 가진다고 말할 수 없다.
그러나 기사마다 제목과 기사 내용이라는 변수를 비교해보면 같은 내용을 다루고 있는지 파악할 수 있다.
가령, 문법적 기능어를 제외하고 가장 많이 등장한 단어를 비교하면 기사의 주제를 비교할 수 있다.
이렇게 비슷한 기사 데이터를 그룹화하는 것을 clustering이라 한다.

예를 들어, 음원 파일을 데이터셋으로 학습한다고 해보자.
음원 파일은 가수의 목소리와 배경 음악이 섞여 있다.
음원 파일을 관찰하면 주파수, 볼륨 등을 변수로 구성할 수 있다.
이때 변수들 간의 관계를 파악하여 가수의 목소리만 찾아내는 것은 non-clustering에 속한다.
```
