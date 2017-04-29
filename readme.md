ⓒ JMC 2017

**SOURCE**  
Coursera ML by Andrew Ng, ML 강의노트 by 박수진

**RESUME**  
`https://www.coursera.org/learn/machine-learning/lecture/GFFPB/gradient-descent-intuition`

---


# 1주차 ::: (4) Paremeter Learning

## 용어 정리

+ gradient descent : (기울기에 대한) 점진적 하강
+ convergence : 한 점으로의 수렴
+ optimum : 점진적 하강으로 찾은 최소화 값
+ `a:=b` : b를 a에 할당한다.
+ alpha : learning rate
+ learning rate : 기울기가 하강한 정도
+ large alpha : 기울기가 급격히 하강함
+ simultaneous update : theta-0과 theta-1을 동시에 구함 (점진적 하강 알고리즘에 적용하는 규칙)


## Gradient Descent (부제: to find parameters minimizing Cost Function)

### Summary

비용 함수를 최소화하는 parameters를 찾기 위해, 비용 함수의 값이 작아지는 곳으로 이동하는 방식을 점진적 하강 알고리즘이라 한다.

### Explain

비용 함수를 최소화하려면 parameters를 조절해야 한다. 이때 3D로 시각화된 그래프를 이용하면 편하다.
시작할 지점을 선택하고 점점 낮은 곳으로 이동한다. 이러한 이동 방식을 점진적 하강이라 한다.
이때 낮은 곳이란 비용 함수의 값을 의미한다.
그리고 점점 낮은 곳으로 이동할 때마다 parameters가 동시에 업데이트 된다.

### Outline

+ 비용 함수를 최소화하는 parameters를 찾으려면 점진적 하강 알고리즘을 사용한다.
+ 점진적 하강 알고리즘이란 시작한 점으로부터 기울기가 가장 낮게 수렴하는 점으로 이동하는 방식을 말한다.
+ 점진적 하강에서 parameters는 동시에 업데이트 된다. (=simultaneous update)

### Visualize

![GradientDescent](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/bn9SyaDIEeav5QpTGIv-Pg_0d06dca3d225f3de8b5a4a7e92254153_Screenshot-2016-11-01-23.48.26.png?expiry=1493596800000&hmac=mMU13dJ3oHYQ_c8d4bqpedu2keNpVibfJkN7GK1xREk)

### Formula

![gradientDescentFromula](http://www.holehouse.org/mlclass/01_02_Introduction_regression_analysis_and_gr_files/Image%20[17].png)

## Gradient Descent - Intuition

@@ resume

---

# 1주차 ::: (3) Model and Cost Function

## 용어 정리

+ model : training data set을 일반화한 함수/방정식
+ model representation : 모델 만들기
+ hypothesis : 가설 함수 (=모델에 대한 함수/방정식)
+ paremeter : 가설 함수의 기울기와 절편
+ error : 오차 (=가설함수와 실제 데이터의 차이)
+ cost function : 모델의 오차를 구하는 함수/방정식


## Model Representation

### Summary

training data set을 일반화하기 위한 가설 함수를 만들면 된다.

### Explain

가설 함수를 만들려면 방정식을 세우면 된다. 단순 회귀모델은 input 변수가 1개이다.
따라서 단순 회귀모델의 가설 함수는 직선을 가진 1차 방정식이 된다.
직선의 방정식을 구하려면 절편 값과 기울기를 알아야 한다.
여기서 기울기를 parameter라고 한다.
parameter에 어떤 값이 들어가느냐에 따라 모델이 달라진다.

### Outline

+ 가설 함수 =  $ h_\theta (x) =  \theta_1 x + \theta_0 $


## Model Parameters

### Summary

좋은 가설 함수를 만들려면, 가설 함수의 오차를 가장 적게 만드는 parameter를 구해야 한다.

### Explain

parameter를 알맞게 정하는 방법은 무엇일까?
가설 함수에 x를 대입한 결과값과 y값의 오차를 최소화하면 된다.

### Outline

+ 가설 함수의 결과값과 실제 y값의 차이를 오차라고 한다.
+ 오차가 적을수록 좋은 parameter이다.
+ 좋은 parameter를 구하려면 오차를 구하는 방정식을 알아야 한다.
+ 가설 함수의 오차를 구하는 방정식을 비용 함수라고 한다.

## Cost Function  (부제: to calculate error)

### Summary

가설 함수의 오차를 측정하는 함수를 비용 함수라고 한다.

### Explain

가설 함수에 x를 대입한 결과값과 y값의 오차를 제곱하고 모두 더한다.
그 후 데이터의 개수로 나눈다. 여기까지는 비용의 평균을 구하는 것이다.
그 후 1/2을 곱한다.
1/2을 곱하는 것은 차후 gradient formula에서 거듭제곱된 cost를 미분할 때 편하게 계산하려는 의도이다.

위와 같이 오차를 거듭제곱해서 평균 비용을 최소화하는 방법을 LSM이라고 한다. LSM은 Least Squared Method로서, 오차를 제곱한 값을 합했을 때 가장 적은 수치를 찾는 방법을 뜻한다. LSM은 regression에서 가장 많이 쓰이는 비용 함수이다.

LSM이 왜 가장 많이 쓰일까?
오차를 제곱하지 않으면 양수와 음수가 존재할 수 있는데,
양수와 음수를 더하면 실제 오차의 값이 상쇄되어 버린다.
이러한 이슈를 방지하는 가장 쉬운 방법으로써 LSM이 쓰인다.

LSM을 사용해서 구한 오차를 LSE라고 한다.
LSE는 Least Squared Error를 뜻한다.

### Outline

+ ![costFunction](http://cfile2.uf.tistory.com/image/2679364956E67C62111446)

+ 비용 함수를 2차원 그래프로 나타내면 등고선 그래프가 그려진다.
+ 비용 함수를 3차원 그래프로 나타내면 아래와 같다.

### Visualize

![visualizingCostFunction](http://www.holehouse.org/mlclass/01_02_Introduction_regression_analysis_and_gr_files/Image%20[14].png)

<!-- ## Minimize Cost Function

### Summary

비용 함수의 값을 최소화하려면, 비용 함수의 값을 미분했을 때 0이 되는 parameter를 구하면 된다. -->

**끝.**

---

# 1주차 ::: (2) Regression

## 용어 정리

+ 선형회귀 : lnear regression
    + supervised learning에 속하고
    + output이 연속적인 값이면서
    + input과 output 사이에 선형관계가 존재할 때 사용한다.


+ 단순회귀 : univariate linear regression
    + 선형회귀 조건을 만족하면서
    + input 변수가 1개일 때 사용한다.
    + 실제로는 거의 사용되지 않고, 이론적인 이해를 돕기 위해 배운다.

> **Note** 더 쉽게 linear regression with one variable이라고 표현해도 된다.


+ 다중회귀 : multivariate linear regression
    + 선형회귀 조건을 만족하면서
    + input 변수가 여러 개일 때 사용한다.
    + 실제로 굉장히 많이 사용되는 학습방법이다.

## 선형회귀는 언제 사용하는가?

### Summary

아래 3가지 조건을 모두 만족할 때 사용한다.

1. 변수가 input과 output으로 나눠질 때 (supervised learning)
2. output 변수가 연속적인 값일 때 (continuous output)
3. input과 output사이에 선형적인 관계가 존재한다고 가정할 때 (linear relationship)


### Outline

+ supervised learning
    + continuous ouput
        + linear relationship between x and y

**끝.**

---

# 1주차 :::: (1) Introduction

## 머신러닝이란 무엇인가?

### Summary

프로그래머가 직접 수많은 규칙을 미리 정해주는 대신 프로그램 자체가 데이터를 통해 스스로 학습하도록 하는 방법을 머신러닝이라 한다.


### Extra

+ 샘플 데이터로 빈 공간을 채우는 것 (from 테리의 딥러닝 토크)
+ 경험적으로 문제를 해결하는 방법을 컴퓨터에 적용한 것 (from 카카오AI리포트)

## Dataset (작성 중)

### Summary

n개의 변수를 갖고 있는 데이터의 집합을 데이터셋이라고 한다.

### Explain

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

### Outline

+ training dataset (=sample dataset)
+ validation dataset
+ test dataset

## Supervised Learning

### Summary

데이터셋의 변수가 input과 output으로 나눠져서, input으로 output을 예측하기 위해 사용하는 학습을 supervised learning이라 한다.

### Explain

supervised learning은 데이터셋의 변수가 input과 output으로 나눠져야 사용할 수 있다. supervised learning은 input으로 output을 예측하기 위해 학습한다. supervised learning은 input이 output에 미치는 영향을 계량화한다.

이때 output이 연속적인 값이라면 regression을 사용하고, output이 분류되어야 하는 범주 값이라면 classification을 사용한다.

예를 들어, 집에 대한 데이터가 있다. 집의 크기, 위치, 건축연도 등이 input 변수이고 집값이 output 변수가 된다. 크기, 위치, 건축연도 등으로 집값을 예측하려면 집값이 연속적인 값이므로 regression을 사용하면 된다.

### Outline

+ 특징 : 데이터셋의 output에 label이 있는 경우

+ supervised learning의 세부 분류
    + regression
    + classification

+ regression vs. classification
    + regression : continuous output
    + classification : discrete output

## Unsupervised Learning이란 무엇인가?

### Summary

데이터셋의 변수가 input과 output으로 나눠지지 않고, 변수를 통해 서로 유사하거나 관계가 있는 데이터를 구분하는 학습을 unsupervised learning이라 한다.

### Explain

unsupervised learning은 데이터셋의 변수가 input과 output으로 나눠지지 않을 때 사용할 수 있다.
unsupervised learning은 변수를 통해 데이터 간의 관계를 파악하기 위해 학습한다.
unsupervised learning은 데이터의 변수들이 서로 유사(similar)하거나 관계(related)가 있는 정도를 계량화한다.

이때, 섞여 있는 데이터셋에서 서로 비슷한 데이터를 그룹화하는 것을 clustering이라 한다. 섞여 있는 데이터셋에서 특정 데이터를 찾아내는 것을 non-clustering이라 한다.

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

### Outline

+ 특징 : 데이터셋의 output에 label이 없는 경우

+ unsupervised learning의 세부 분류
    + clustering : grouping
    + non-clustering: identifying


**끝.**

---
