# kdata-biganl

## Anaconda Upgrade
```
$ python -m pip install –-upgrade pip
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
model = DecisionTreeRegressor(
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

## SVM - Classification
```
from sklearn.svm import SVC
model = SVC(
    C=10.0
   ,gamma=10000
   ,kernal='rbf'
   ).fit(arrTrainFeatures, arrTrainLabels)
pred = model.predict(arrTestFeatures)
```

## SVM - regression
```
from sklearn.svm import SVR
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3< degree=2)
pred_rbf = svr_rbf.fit(featuresTrain, labelTrain).predict(featuresTrain)
pred_lin = svr_lin.fit(featuresTrain, labelTrain).predict(featuresTrain)
pred_poly = svr_poly.fit(featuresTrain, labelTrain).predict(featuresTrain)
```

### Random Forest - Classifacation
```
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier().fit(arrTrainFeatures, arrTrainLabels)
predTestLabels = model.predict(arrTestFeatures)
```

### Random Forest - Regression
```
from sklearn.ensemble import RandomForestRegressor
model = RandomForestClassifier().fit(arrTrainFeatures, arrTrainLabels)
predTestLabels = model.predict(arrTestFeatures)
```

### k-NN - Classifacation
```
from sklearn import neighbors
model = neighbors.KNeighborsClassifier(
    n_neighbors = n_neighbors
   ,weights='distance'
    ).fit(dfTrain[['X3','X4','X5']], dfTrain['Y2'])
predTestLabels = model.predict(dfTest[['X3','X4','X5']])
```

### k-NN - Regression
```
from sklearn import neighbors
model = neighbors.KNeighborsRegressor(
    n_neighbors = n_neighbors
   ,weights='distance'
   ,p=2
    ).fit(dfTrain[['X3','X4','X5']], dfTrain['Y1'])
predTestLabels = model.predict(dfTest[['X3','X4','X5']])
```

## k-NN - Clustering
```
from sklearn import neighbors
model = neighbors.KNeighborsRegressor().fit(dfTrain[['X1','X2','X3']], dfTrain['Y1'])
distances, indices = model.kneighbors(dfTest[['X1','X2','X3']])
```

## glm - ridge
```
import sklearn import linear_model as lm
glm_ridge = lm.Ridge(alpha=0)
            .fit(
                dfTrain[['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11']]
               ,dfLabel['Y1']
predLabelTrain = glm_ridge.predict(dfTrain[['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11']])
r2 = r2_score(dfTrain['Y'},predLabelTrain)
ridge_coef = Series(
    glm_ridge.coef_
   ,dfTrain[['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11']].columns
   ).sort_values()
```
## glm - lasso
```
import sklearn import linear_model as lm
import sklearn.metrics import r2_score
glm_lasso = lm.Lasso(alpha=0)
            .fit(
                dfTrain[['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11']]
               ,dfLabel['Y1']
predLabelTrain = glm_lasso.predict(dfTrain[['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11']])
r2 = r2_score(dfTrain['Y'},predLabelTrain)
ridge_coef = Series(
    glm_ridge.coef_
   ,dfTrain[['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11']].columns
   ).sort_values()
```
## glm - elastic net
```
import sklearn import linear_model as ElasticNet
import sklearn.metrics import r2_score
glm_ridge = lm.Ridge(alpha=0.1, l1_ratio=0.7)
            .fit(
                dfTrain[['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11']]
               ,dfLabel['Y1']
predLabelTrain = glm_ridge.predict(dfTrain[['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11']])
r2 = r2_score(dfTrain['Y'},predLabelTrain)
ridge_coef = Series(
    glm_ridge.coef_
   ,dfTrain[['X1','X2','X3','X4','X5','X6','X7','X8','X9','X10','X11']].columns
   ).sort_values()
```

## kmeans - clustering
```
from sklearn.cluster import KMeans
kmeans = KMeans(
    n_cluster = 3
   ,random_state = 0
   ).fit(dfTrain[['V9','V10','V11']])
```

## Mahalanobis

## dtw
```
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
x = np.array([
    [1,1]
   ,[2,2]
   ,[3,3]
   ,[4,4]
   ,[5,5]
   ])
y = np.array([
    [0,0]
   ,[2,2]
   ,[3,3]
   ,[4,4]
   ,[5,5]
   ])
distance, path = fastdtw(x,y,dist=euclidean)
print(distance)
```
## word2vec
- https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec

## tensorflow - logistic regression
```
import tensorflow as tf
x = tf.placeholder(tf.float32, shape=[row,col_x], name="x-feature")
y = tf.placeholder(tf.float32, shape=[row,col_y], name="y-label")
W1 = tf.Variable(tf.random_uniform([col_x, 1],-1,1), name="Weight1")
b1 = tf.Variable(tf.zeros([1], name="bias1")
Hypothesis = tf.sigmoid(tf.matmul(x,W1) + b1)
cost = tf.reduce_mean(-y*tf.log(Hypothesis) - (1-y)*tf.log(1-Hypothesis))
learning_rate = 0.1
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
init = tf.global_variable_initializer()
with tf.Session() as sess:
    sess.run(init)
    for step in range(10001):
        step_opt = sess.run(optimizer, feed_dixt({x:feature, y=label})
        if step % 1000 == 0:
            step_cost = sess.run(cost, feed_dict = {x:feature, y:label})
            print(step, step_cost)
        predicted_label = sess.run(Hypothesis, feed_dict={x:feature})
        print(predicted_label)
        print('Weight1 :{} Bias 1 {}'.format(sess.run(W1), sess.run(b1)))
        correct_prediction = tf.equal(tf.round(Hypothesis), label)
        accuracy_rate = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print('accuracy rate : ', sess.run(accuracy_rate, feed_dict={x:feature, y:label})
```
## tensorflow - Single Layer Perceptron

## tensorflow - Multi Layer Perceptron

## tensorflow - CNN

## tensrflow - RNN



