ⓒ JMC 2017

**SOURCE**  
Coursera ML by Andrew Ng, ML 강의노트 by 박수진

**NOTE-TAKING**  
수학 기호 : ㅎ/ㄷ + 한자키

**RESUME**  
2주차 `https://www.coursera.org/learn/machine-learning/lecture/2DKxQ/normal-equation`

---

**목차**  

<ul>
<li><a href="#1주차--1-introduction">1주차 ::: (1) Introduction</a></li>
<li><a href="#1주차--2-regression">1주차 ::: (2) Regression</a></li>
<li><a href="#1주차--3-model-and-cost-function">1주차 ::: (3) Model and Cost Function</a></li>
<li><a href="#1주차--4-parameter-learning">1주차 ::: (4) Parameter Learning</a></li>
<li><a href="#1주차--5-linear-algebra-review">1주차 ::: (5) Linear Algebra Review</a></li>
<li><a href="#2주차--1-multivariate-linear-regression">2주차 ::: (1) Multivariate Linear Regression</a></li>
<li><a href="#"></a></li>
<li><a href="#"></a></li>
<li><a href="#"></a></li>
<li><a href="#"></a></li>
<li><a href="#"></a></li>
</ul>

---

# 2주차 ::: (6) Computing Parameters Analytically

## 용어 정리


## Summary


## Explain



---

# 2주차 ::: (5) Features and Polynomial Regression (부제: data에 fit하게 feature를 올바르게 선택하는 방법)

## 용어 정리

+ `polynomial regression` : feature의 형태를 다양하게 만들어야 하는 다항식 회귀

## Summary

데이터에 fit한 모델을 만들려면 feature들을 각각 다르게 design해야 한다.

## Explain

+ 때때로 새로운 feature를 정의했을 때 더 나은 모델을 얻을 수 있다.
+ 여러 가지 feature를 다양하게 만들어야 하는 회귀를 polynomial regression이라고 한다.

> **Note:** polynomial regression은 '다항식 회귀'를 뜻한다.

+ hypothesis function이 반드시 linear할 필요는 없다.
+ 데이터에 fit하려면 다양한 형태를 가질 수 있다.
+ 예를 들면 2차 함수, 3차 함수, 제곱근 함수를 활용할 수 있다.

![polynomialRegression](https://github.com/datalater/machine-learning/blob/master/images/polynomialRegression.PNG?raw=true)

+ 위 그래프에 대한 설명은 다음과 같다.
+ 초기에 위쪽으로 볼록하게 휘어진 x점에 fit하려면 2차 함수를 떠올릴 수 있다.
+ 그런데 2차 함수는 나중에 감소하는 형태이므로 적절하지 않다.
+ 그러므로 3차 함수를 고려할 수 있다.
+ 3차 함수 꼴로 나타내었을 때 새로운 feature `x_2`와 `x_3`가 추가된다.
+ 또 제곱근 형태로 feature를 만들 수도 있다.
+ 여기서 주의할 점은 feature의 차수가 다르기 때문에 feature scaling이 중요해진다는 것이다.
+ 가령, 3차 함수 꼴로 나타낸다면 각 feature의 범위는 다음과 같다.
    + x_1의 범위 : 1~`10^3`
    + x_2의 범위 : 1~`10^6`
    + x_3의 범위 : 1~`10^12`

## Quiz

![polynomialRegression-Quiz](https://github.com/datalater/machine-learning/blob/master/images/polynomialRegression-quiz2.png?raw=true)

**끝.**

---

# 2주차 ::: (4) Gradient Descent in Practice II - Learning Rate


## 용어 정리

+ `debugging` : gradient descent가 제대로 작동하고 있는지 확인하는 방법
+ `automatic convergence test` : gradient descent를 한 번 iterate했을 때 cost function의 값이 `10^-3`보다 적게 감소하면 convergence라고 선언한다.
    + 단, 반드시 `10^-3`일 필요는 없고 이처럼 작은 값이 기준이 된다는 것이다.
    + 그러나 `10^-3`과 같은 작은 값을 정하는 기준이 명확하지 않기 때문에 Andrew Ng 교수는 반복 횟수에 따른 cost function의 값을 표현한 그래프를 통해 convergence 여부를 확인하는 것을 선호한다.

## Summary

gradient descent가 제대로 작동하고 있는지 알려면 iterate할 때 마다 cost function의 값이 줄어드는지 그래프로 확인해보면 된다.
그리고 cost function의 값이 빠르게 converge하도록 alpha 값의 범위를 not too small and not too large한 범위에서 조절해야 한다.

## Explain

+ `Debugging` : gradient descent가 제대로 작동하고 있는지 확인하는 방법
+ gradient descent가 제대로 작동하고 있다면 iteration을 반복할수록 cost function의 value가 줄어들어야 한다.
+ 이를 함수의 그래프로 나타낸다면, cost fuction의 값이 줄어든다면 gradient descent가 제대로 작동하고 있는 것이고, 줄어들다가 평평해지는 지점이 생긴다면 그곳이 바로 convergence가 될 것이다.
+ 따라서 다음과 같은 함수의 그래프를 떠올릴 수 있다.

![debuggingGradientDescent]()

+ 또, debugging을 위해 cost function의 값이 converge하고 있는지 테스트해봐도 된다.
+ 이를 automatic convergence test라고 한다.

> **Note:** 그러나 `## 용어 정리`에 있는 내용처럼 Ng 교수는 함수의 그래프를 통해 debugging 하는 것을 더 선호한다.

+ alpha가 너무 작으면 : 느리게 수렴한다
+ alpha가 너무 크면 : 반복해도 감소하지 않거나 수렴하지 않는다

+ alpha 값을 선택하는 방법 :
    + too small한 alpha와 too large한 alpha를 찾고 그 사이에서 조절한다.
    + cost function의 값을 빠르게 감소시키도록 alpha 값을 비율을 늘린다.
    + ex. 0.001 > 0.003 > 0.01 > 0.03 > 0.1 > 0.3 > 1

**끝.**


---

# 2주차 ::: (3) Gradient Descent in Practice I - Feature Scaling

## Summary

모든 feature가 비슷한 범위에 있으면 gradient descent가 더 빠르게 수렴하는 데에 도움이 된다.
그래서 Feature Scaling을 한다.

## Explain

+ 두 가지 feature가 있을 때 x1의 범위가 x2의 범위보다 훨씬 크다면,
+ countour plot이 아래처럼 매우 길쭉한 타원형(skewed oval)이 된다.
+ 그렇게 되면 gradient descent가 global minimum까지 도달하는 시간이 길어지게 된다.

![BeforeFeatureScaling](https://github.com/datalater/machine-learning/blob/master/images/BeforeFeatureScaling.png?raw=true)

+ 그러나 범위를 조절하면 contour plot이 좀 더 원형(circle)에 가깝게 된다.
+ 원형에 가까운 상태일 수록 gradient descent가 global minimum까지 도달하는 시간이 짧아진다.
+ 이렇게 feature들의 범위를 서로 비슷하게 조절하는 것을 feature scaling이라 한다.

![AfterFeatureScaling](https://github.com/datalater/machine-learning/blob/master/images/AfterFeatureScaling.png?raw=true)

+ Feature Scaling : 모든 변수의 범위를 대략 `-1<=x_i<=1` 로 맞추는 것
+ 단, 비슷하면 좋은 것이지 특정 변수의 범위가 반드시 -1과 1사이이거나 변수끼리 범위가 똑같을 필요는 없다.
    + `-3<=x_1<=3` (o)
    + `-1/3<=x_2<=1/3` (o)
    + `-100<=x_3<=100` (x)
    + `-0.0001<=x_4<=0.0001` (x)

> **Note:** 단, `x_0=1` 이다.

+ Feature Scaling의 한 가지 방법 : mean normalization (정규화)
+ 모든 feature의 평균을 0으로 만들기 위해 `x_i`를 `(x_i-mu_i)/(max-min)`로 대체한다.
+ 주의할 점은 feature scaling이 반드시 정확하게 계산할 필요는 없다는 것이다.
    + 가령 x_1을 정규화했을 때 분모가 5이고 x_2를 정규화했을 때 분모가 4라면, 두 feature의 범위를 비슷하게 만들어주기 위해 x_2를 정규화할 때 분모를 5로 변경해도 괜찮다.


**끝.**

---

# 2주차 ::: (1) Multivariate Linear Regression

## 용어 정리

+ m : training example의 수
+ n : feature의 수
+ $ x^{(i)} $ : i번째 training example의 features를 원소로 하는 벡터
    + i는 training example의 index를 뜻함.
    + "테이블의 i번째 줄"을 보라는 뜻
    + i번째 training example의 모든 features를 가리키며, 벡터로 표현함.
+ $ x^{(i)}_j $ : $ x^{(i)} $ 벡터의 j번째 원소
    + j는 $ x^{(i)} $ 벡터의 index를 뜻함.


## Multiple Features

### Summary

output 변수 y를 예측하기 위해 input 변수가 여러 개 있는 것을 multiple features라고 한다.

> **Note** 여기서 features와 variables를 같은 뜻으로 봐도 무방하다. multiple variables 또는 multivariate는 우리말로 "다변량"으로 번역하기도 한다.

### Explain

다중회귀의 가설 함수는 n개의 `x`와 n+1개의 `theta`로 이루어진다. 이때 계산을 편하게 하기 위해 `x_0=1`로 정의한다. 그러면 아래 그림과 같이 벡터 `x`와 벡터 `theta`를 표현할 수 있다.

![multivariate_hypothesis_function](https://wikidocs.net/images/page/7639/muti104.PNG)

가설 설함수는 `theta`와 `x`의 곱으로 이루어진다. 이를 한번에 packing 해서 표현하기 위해 벡터 `theta`를 전치행렬(`theta^T`)로 전환하면 행렬의 곱셈 형태(`theta^T * x`)로 나타낼 수 있다.

### Outline

+ multiple features의 가설 함수는 행렬의 곱셈으로 나타낼 수 있다.
+ multiple features의 가설 함수의 parameters는 `theta_0`부터 `theta_n`까지 있다.
+ 이 수업부터 성질이 같은 여러 개의 원소는 벡터로 표현하게 될 것이다.
+ multiple features의 가설 함수의 parameters는 (n+1)-dimensional vector이다.
+ multiple features의 가설 함수의 parameters는 어떻게 찾을까?

## Cost Function for Multiple Variables

### Summary

multiple features의 비용 함수는 변수가 1개일 때의 비용 함수와 같은 꼴이지만, parameters를 벡터화해서 나타낸다.

> **Note** 벡터화 : vectorize

### Explain

![CostFunctionForMultipleVariables](http://www.holehouse.org/mlclass/04_Linear_Regression_with_multiple_variables_files/Image.png)

위 식을 vectorize 하면, 체크박스의 식과 같아진다.

![CostFunctionForMultipleVariables_vectorized](https://github.com/datalater/machine-learning/blob/master/images/CostFunctionForMultipleVariables_vectorized.png?raw=true)

### Outline

+ multiple features의 비용 함수는 변수가 1개일 때의 비용 함수와 같은 꼴이지만, parameters를 벡터화해서 나타낸다.
+ multiple features의 비용 함수의 최소값도 점진적 하강 알고리즘을 적용한다.

## Gradient Descent for Multiple Variables

### Summary

변수가 하나일 때의 점진적 하강 알고리즘과 같은 꼴이지만, n개의 features에 대해 반복한다는 점이 다르다.

### Explain

![GradientDescentForMultipleFeatures](http://www.holehouse.org/mlclass/04_Linear_Regression_with_multiple_variables_files/Image%20[3].png)

n개의 features를 가진 점진적 하강 알고리즘은 위 그림과 같다. 위 식을 n개의 features, 즉 n개의 parameters별로 적용해야 한다.

### Outline

+ multiple features의 점진적 하강 알고리즘은 변수가 1개일 때의 점진적 하강 알고리즘과 같은 꼴이지만, n개의 features에 대해 반복한다는 점이 다르다.

**끝.**

---

# 1주차 ::: (5) Linear Algebra Review

## Matrices and Vectors

## 용어 정리

+ **matrix** : 숫자로 구성된 직사각형 배열 (= 2차원 배열 = 행렬)
+ matrices : matrix의 복수형
+ dimension of matrix : 행렬의 차원. row 개수 * column 개수. R로 표현.
+ **vector** : 행렬의 특별한 케이스. column이 1개인 행렬. N by 1 matrix
+ n-dimensional vector : row가 n개인 벡터
+ **scalar** : 벡터나 행렬처럼 배열이 아니라 원소가 하나(single value)인 객체를 가리키는 말. 따라서 벡터나 행렬의 원소를 scalar라고 칭할 수 있음.
+ real number : scalar와 같은 뜻이라고 봐도 무방
+ **matrix vs. vector vs. scalar** : scalar가 2차원 배열로 구성되면 matrix이고, scalar가 2차원 배열이면서 column이 1개이면 vector가 됨.
+ 대문자 변수 : 주로 matrix를 가리킴. ex. `Y`
+ 소문자 변수 : 주로 vector를 가리킴. ex. `y`
+ 1-indexed : 1부터 시작하는 index
+ 0-indexed : 0부터 시작하는 index
+ identity matrix : 단위 행렬. 곱셈에서 교환법칙이 성립되게 만드는 정사각 행렬

## Matrix-Vector Multiplication

### Summary

회귀 알고리즘에서 하나의 가설 함수의 `(x, h(x))` 값을 구할 때, 간단하고 효율적으로 코드를 작성하기 위해 "**행렬과 벡터의 곱셈**"을 사용한다.

> **Note** 단순회귀 알고리즘에서 데이터는 행렬로 나타내고 방정식의 parameters는 벡터로 나타낸다. 이 둘을 곱하면 가설 함수의 예측 값을 계산할 수 있다.

### Visualize

+ 집 크기로 집값을 예측하는 회귀식에 행렬과 벡터의 곱셈을 적용하는 방법은 다음과 같다.

![matrix_vector_multiplication01](https://github.com/datalater/machine-learning/blob/master/images/matrix_vector_multiplication01.jpg?raw=true)

+ 이러한 연산을 다음과 같이 일반화할 수 있다.

![matrix_vector_multiplication02](https://github.com/datalater/machine-learning/blob/master/images/matrix_vector_multiplication02.jpg?raw=true)

+ 가설 함수의 값 (prediction) = 데이터 행렬 * parameters

### Outline

+ `m by n` matrix * `n by 1` vector = `m by 1` vector
+ 단순회귀에서 데이터는 행렬로 나타내고 parameters는 vector로 나타낸다.

## Matrix-Matrix Multiplication

### Summary

회귀 알고리즘에서 "**여러 개**"의 가설 함수의 `(x, h(x))` 값을 구할 때, 간단하고 효율적으로 코드를 작성하기 위해 "**행렬과 행렬의 곱셈**"을 사용한다.

> **Note** 데이터의 원소가 m개이고 가설 함수가 n개 있을 때, 이를 한번에 계산할 수 있다.

### Visualize

+ 데이터 4개, 가설 함수 3개라면 12개의 예측값(prediction)을 구해야 한다.

+ 아래와 같이 행렬과 행렬의 곱셈을 이용하면 12개의 예측값을 한번에 효율적으로 구할 수 있다.

![matrix_matrix_multiplication01](https://github.com/datalater/machine-learning/blob/master/images/matrix_matrix_multiplication01.jpg?raw=true)

### Outline

+ 데이터와 가설함수의 parameters를 각각 행렬로 나타내서 곱한다.
+ 수많은 연산을 한번에 packing 하기 위해 행렬의 곱셈을 사용한다.

## Matrix Multiplication Properties

### Summary

행렬의 곱셈은 실수의 곱셈과 달라서, 교환법칙이 성립되지 않는다. 하지만 단위행렬을 활용하면 곱셈의 교환법칙이 성립된다.


## Inverse and Transpose

### Summary

어떤 행렬의 역행렬이 없다면 그 행렬의 원소는 0에 가깝다는 뜻이며, 알고리즘을 계산할 때 행과 열을 바꾸는 전치행렬을 사용해야 할 때가 있을 것이다. (추측)

> **Note** 역행렬 (inverse matrix), 전치행렬 (transpose matrix)

### Explain

행렬 A와 곱했을 때 그 결과가 단위 행렬이 나오는 행렬을 행렬 A의 역행렬이라 한다. 그런데 영행렬은 역행렬이 없다. 머신러닝에서 가져가야 할 인사이트는 어떤 행렬의 역행렬이 없다면 그 행렬의 원소는 0에 가깝다는 것이다. 역행렬이 없는 행렬을 "singular" 또는 "degenerate"라고 한다.

행렬 A의 행과 열을 뒤바꾼 것을 행렬 A의 전치행렬이라 한다. 전치행렬을 만드는 관점은 크게 3가지가 있다.
첫번째, row와 column을 한꺼번에 이동하는 관점. 첫 번째 row의 값들이 첫 번째 colum의 값들로 변경되고, 두 번째 row의 값들이 두 번째 column의 값들로 변경된다.  
두번째, row와 column으로 이루어진 주소를 하나씩 이동하는 관점. (1,2)의 값이 (2,1)의 값으로 변경된다.  
세번째, 대각선을 기준으로 값을 대칭시키는 관점. (k, k)의 값은 그대로 두고 나머지 값을 대각선을 기준으로 대칭시킨다.

지금까지 배운 선형대수의 행렬과 벡터의 개념은 이번 머신러닝 코스에서 계속 사용될 것이다. 강력한 알고리즘을 끌어내기 위해서는 이러한 선형대수 개념이 필요하다는 것을 기억해두자.

**끝.**

---

# 1주차 ::: (4) Parameter Learning

## 용어 정리

+ gradient descent : (기울기에 대한) 점진적 하강
+ convergence : 한 점으로의 수렴
+ optimum : 점진적 하강으로 찾은 최소화 값
+ `a:=b` : b를 a에 할당한다.
+ `alpha` : learning rate
+ learning rate : 기울기가 하강한 정도
+ large `alpha` : 기울기가 급격히 하강함
+ simultaneous update : `theta-0`과 `theta-1`을 동시에 구함 (점진적 하강 알고리즘에 적용하는 규칙)
+ differential : 접선의 기울기 (= 순간변화율 = 미분 = derivative)
+ partial derivative : 편미분
+ minimum : derivative가 0인 곳
+ batch gradient descent : 점진적 하강은 한 번 하강할 때마다 모든 데이터셋 무리(the entire batch of training examples)를 이용한다는 것을 강조하는 말로 점진적 하강과 같은 말이다.

> **Note** derivative와 differential은 다른 개념이지만, 본 수업에서는 이해를 돕기 위해 같은 것으로 간주한다.


## Gradient Descent (부제: to find parameters minimizing cost function)

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

![GradientDescent](http://www.holehouse.org/mlclass/01_02_Introduction_regression_analysis_and_gr_files/Image%20[16].png)

### Formula

+ 수렴하는 점(convergence)을 찾을 때까지 아래 공식을 반복한다.

![gradientDescentFromula](http://www.holehouse.org/mlclass/01_02_Introduction_regression_analysis_and_gr_files/Image%20[17].png)

+ `theta` 값은 아래와 같이 동시에 업데이트한다.

![simultaneous-update](http://www.holehouse.org/mlclass/01_02_Introduction_regression_analysis_and_gr_files/Image%20[19].png)

## Gradient Descent Formula ::: (1) derivative

### Summary

점진적 하강 알고리즘은 `alpha`와 편미분항으로 구분되는데, `alpha`는 한 번에 하강하는 정도를 뜻하고 편미분항은 이동할 방향과 크기를 결정한다.

### Explain

시작점으로부터 점진적 하강 공식을 적용해보자. `theta-j`값은 시작점으로부터 이동하게 된다.
이때 `alpha`는 항상 0보다 크므로, 이동 방향은 편미분항의 값에 따라 결정된다.

편미분항은 접선의 기울기를 의미하는데, 접선의 기울기가 positive하다면 `alpha`와 편미분항의 곱이 양수가 되어 `theta-j`는 시작점보다 값이 작아지게 된다.
시작점보다 작아진다는 것은 그래프 상에서 `theta-j`축으로부터 왼쪽으로 이동한다는 뜻이다. 점점 minimum에 가까워진다.

반대로 접선의 기울기(=편미분항)가 negative하다면 `theta-j`는 시작점보다 값이 커지게 된다. 그래프 상에서 `theta-j`축으로부터 오른쪽으로 이동한다. 점점 minimum에 가까워진다.

### Outline

+ 편미분항 값을 구하려면 시작점의 접선의 기울기를 구하면 된다.
+ 접선의 기울기가 positive하다면, `theta`값은 작아진다.
+ 접선의 기울기가 negative하다면, `theta`값은 커진다.

## Gradient Descent Formula ::: (2) alpha

### Summary

`alpha` 값에 따라 minimum까지 도착하는데 오래 걸릴 수도 있고, 아예 멀어지거나 발산할 수도 있다.

### Explain

`alpha` 값이 너무 작으면 편미분항을 곱해도 값이 미미하므로 아주 조금씩 이동하게 되고 결국 minimum까지 도착하는데 오래 걸린다.

`alpha` 값이 너무 크면 편미분항을 곱했을 때 값이 너무 커져서 minimum으로 수렴하지 못하거나 심지어 발산하게 된다.

### Visualize

![alpha](https://wikidocs.net/images/page/7635/linreg701.PNG)

### Outline

+ `alpha`는 항상 0보다 크다.
+ `alpha`를 너무 작게 설정하면 minimum까지 도착하는데 오래 걸린다.
+ `alpha`를 너무 크게 설정하면 minmum으로부터 멀어지거나 심지어 발산하게 된다.

## Gradient Descent Formula ::: (3) 시작점

### Summary

시작점에 따라 global minimum이 아니라 local minimum으로 수렴할 수도 있다.

### Explain

시작점에서 점진적 하강을 하다보면 편미분항이 0이 되는 시점을 발견할 수 있다.
그러나 이 시점은 local minimum일뿐 반드시 global minimum이 라고 할 수는 없다.
더 낮은 minimum이 있다고 하더라도, 한 번 편미분항이 0이 되면 더 이상 이동하지 않기 때문이다. 아래 그림을 참조해보자.

### Visualize

![globalMinimum](https://wikidocs.net/images/page/7635/linreg702.PNG)

## Gradient Descent For Linear Regression (부제: Apply Gradient Descent to Minimize Cost Function of Linear Regression)

### Summary

선형 회귀의 비용 함수에 점진적 하강을 적용하면 유일한 global optima를 찾을 수 있다.
global optima에 해당하는 parameters를 적용하면 선형 회귀 모델을 만들 수 있다.

### Explain

지금까지의 내용을 요약하면 다음과 같다.
샘플 데이터가 잘 맞는(fit) 선형 회귀 모델을 만들려면, 샘플 데이터와 가설 함수의 오차가 최소화되어야 한다.
그런데 오차는 모델의 parameters를 어떻게 잡느냐에 따라 달리진다.
parameters에 따른 오차를 구하려면 비용 함수를 적용해야 한다.
선형 회귀에서는 LSM을 비용 함수로 적용한다.
우리는 오차가 가장 적은 모델을 구해야 하므로 비용 함수의 최소값을 알아야 한다.
비용 함수의 최소값을 알려면 점진적 하강 알고리즘을 적용해야 한다.

선형 회귀의 비용 함수는 밥그릇 같은 볼록 함수 모양을 띈다.
볼록 함수에 점진적 하강 알고리즘을 적용하면 단 한 개의 optimum을 구할 수 있다.
이러한 global optimum에 해당하는 parameters를 구하면 우리가 원하는 선형 회귀 모델을 완성할 수 있다.

> **Note** 처음에는 임의로 설정한 parameter에서 출발하여 iteration이 거듭될수록 점점 정확한 hypothesis가 되는 것이다. 또한, Linear regression cost function은 convex이므로 항상 global optima에 수렴한다. (출처: https://wikidocs.net/7635)

### Visualize

+ 비용 함수에 점진적 하강 알고리즘을 적용했을 때 시작점에 따라 local optima가 달라질 수 있다.
+ 그러나 선형 회귀의 비용 함수는 딱 1개의 global optimum을 가진다.
+ 선형 회귀의 비용 함수는 항상 다음과 같이 밥그릇(bowl) 모양을 띈다.
+ 볼록 함수(convex function)라고 부른다.

![visualizingCostFunction](http://www.holehouse.org/mlclass/01_02_Introduction_regression_analysis_and_gr_files/Image%20[14].png)

+ 선형 회귀 모델이 2차 방정식일 때, 비용 함수에 점진적 하강을 적용한 모습은 다음과 같다.
+ 파란색 점(`theta`)이 바깥 원에서 안쪽 원으로 이동하면서 편미분항이 0이 되는 곳(global minimum)을 찾아간다.

![visualizingCostFunction-quadratic](https://d3c33hcgiwev3.cloudfront.net/imageAssetProxy.v1/xAQBlqaaEeawbAp5ByfpEg_24e9420f16fdd758ccb7097788f879e7_Screenshot-2016-11-09-08.36.49.png?expiry=1493596800000&hmac=p7Sphj6I1frNmGOWh-U4bwvlDN2znIQg2sVaSb8giUE)

> **Note**  optima는 복수이고 optimum은 단수이다.

**끝.**

---

# 1주차 ::: (3) Model and Cost Function

## 용어 정리

+ model : training data set을 일반화한 함수/방정식
+ model representation : 모델 만들기
+ hypothesis : 가설 함수 (=모델에 대한 함수/방정식)
+ parameter : 가설 함수의 매개변수 (=함수의 기울기 또는 절편)
+ error : 오차 (=가설 함수와 실제 데이터의 차이)
+ cost function : 모델의 오차를 구하는 함수/방정식


## Model Representation

### Summary

training data set을 일반화하기 위한 가설 함수를 만들면 된다.

### Explain

가설 함수를 만들려면 방정식을 세우면 된다. 단순 회귀모델은 input 변수가 1개이다.
따라서 단순 회귀모델의 가설 함수는 직선을 가진 1차 방정식이 된다.
직선의 방정식을 구하려면 절편 값과 기울기를 알아야 한다.
여기서 기울기나 절편을 parameter라고 한다.
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

![costFunction](http://cfile2.uf.tistory.com/image/2679364956E67C62111446)

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

+ 선형회귀 : linear regression
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
3. input과 output 사이에 선형적인 관계가 존재한다고 가정할 때 (linear relationship)


### Outline

+ supervised learning
    + continuous ouput
        + linear relationship between x and y

**끝.**

---

# 1주차 ::: (1) Introduction

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
