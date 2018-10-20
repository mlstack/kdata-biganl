# kdata-biganl

## Anaconda Upgrade
```
$ python -m pip install –upgrade pip
```



## numpy
- 메모리에 올라온 데이터 배열(array)의 processing
    - 가로세로바꿈
    - 정렬
    - 사칙연산
    - 파생변수만들기
    - 배열재생성
- csv, txt, parquet(local, hadoop) 등의 파일을 pandas의 기능을 이용해서 만든 dataframe을 numpy 배열로 바꾸어서 data processing 진행
- machine learning에서 데이터를 입력받을 때 numpy format


## Decision Tree - Classification

```python
from sklearn.tree import DecisionTreeClassifier
# model = DecisionTreeClassifier()
model = DecisionTreeClassifier(
    criterion = 'gini' # ['gini', 'entropy']
   ,max_depth = 2)
   .fit(dfTrainFeatures, dfTrainLabels)
dfTestLabelsPred = model.prediction(dfTestFeatures)
```
## Decision Tree - Regression

```python
from sklearn.tree import DecisionTreeRegressor
# model = DecisionTreeRegressor()
model = DecisionTreeClassifier(
    criterion = 'mse' # mean squared error
   ,max_depth = 2)
   .fit(dfTrainFeatures, dfTrainLabels)
dfTestLabelsPred = model.prediction(dfTestFeatures)
```

## naive-bayes : Bernoulli - classification
```
from sklearn.naive_bayes import BernoulliNB
model = BernoulliNB(
    alpha=1.0
   ,binarrize = 0.0
   ,class_prior=None
   ,fit_prior=True
   ).fit(arrTrainFeatures, arrTrainLabels)
pred = model.predict(arrTestFeatures)
```
## naive-bayes : Multinomial - classification
```python
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB().fit(arrTrainFeatures, arrTrainLabels)
pred_proba = model.predict_broba(arrTestFeatures)
pred = model.predict(arrTestFeatures)
```



## naive-bayes : GaussianNB - classification
```python
from sklearn.naive_bayes import GaussianNB
model = GaussianNB().fit(arrTrainFeatures, arrTrainLabels)
pred_proba = model.predict_broba(arrTestFeatures)
pred = model.predict(arrTestFeatures)
```
