---
layout: splash
title: "Kaggle: Bike Sharing Demand"
date:   2019-10-22 11:13:00 -0500
---

```python
import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

mpl.rcParams['axes.unicode_minus'] = False

import warnings
warnings.filterwarnings('ignore')
```


```python
train = pd.read_csv('../DATA/train.csv', parse_dates=['datetime'])
train.shape
```




    (10886, 12)




```python
test = pd.read_csv('../DATA/test.csv',parse_dates=['datetime'])
test.shape
```




    (6493, 9)




```python
test.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }
    
    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>season</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>weather</th>
      <th>temp</th>
      <th>atemp</th>
      <th>humidity</th>
      <th>windspeed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011-01-20 00:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>11.365</td>
      <td>56</td>
      <td>26.0027</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-01-20 01:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>13.635</td>
      <td>56</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-01-20 02:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>13.635</td>
      <td>56</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-01-20 03:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>12.880</td>
      <td>56</td>
      <td>11.0014</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011-01-20 04:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>12.880</td>
      <td>56</td>
      <td>11.0014</td>
    </tr>
  </tbody>
</table>
</div>




```python
train["year"] = train["datetime"].dt.year
train["month"] = train["datetime"].dt.month
train["hour"] = train["datetime"].dt.hour
train["dayofweek"] = train["datetime"].dt.dayofweek
train.shape
```




    (10886, 19)




```python
test["year"] = train["datetime"].dt.year
test["month"] = train["datetime"].dt.month
test["hour"] = train["datetime"].dt.hour
test["dayofweek"] = test["datetime"].dt.dayofweek
test.shape
```




    (6493, 16)



**Visualization**


```python
# windspeed : many zeros
fig, axes = plt.subplots(nrows=2)
fig.set_size_inches(18,10)

plt.sca(axes[0])
plt.xticks(rotation=30, ha='right')
axes[0].set(ylabel='Count', title="train windspeed")
sns.countplot(data=train, x="windspeed", ax=axes[0])

plt.sca(axes[1])
plt.xticks(rotation=30, ha='right')
axes[1].set(ylabel='count', title='test windspeed')
sns.countplot(data=test, x='windspeed', ax=axes[1])



```




    <matplotlib.axes._subplots.AxesSubplot at 0xdf0e630>




![png](output_7_1.png)


**speed zero -> Random Forest estimated value**
평균을 구해서 일괄적으로 넣어줄 수도 있지만, 예측의 정확도를 높이기 위해서


```python
# average windspeed
#train.loc[train['windspeed']==0, 'windspeed'] = train['windspeed'].mean()
#test.loc[train['windspeed']==0, 'windspeed'] = train['windspeed'].mean()
# 풍속이 0인것과 아닌 것의 세트를 나누어 준다.

trainWind0 = train.loc[train['windspeed'] == 0]
trainWindNot0 = train.loc[train['windspeed'] != 0]
print(trainWind0.shape)
print(trainWindNot0.shape)
```

    (1313, 19)
    (9573, 19)



```python
from sklearn.ensemble import RandomForestClassifier

def predict_windspeed(data):
    
    # 풍속이 0인것과 아닌 것을 나누어 준다.
    dataWind0 = data.loc[data['windspeed'] == 0]
    dataWindNot0 = data.loc[data['windspeed'] != 0]
    
    #풍속을 예측할 피처를 선택한다.
    wCol = ["season", "weather", "humidity", "month", "temp", "year", "atemp"]

    #풍속이 0이 아닌 데이터들의 타입을 스트링으로 바꿔준다
    dataWindNot0["windspeed"] = dataWindNot0["windspeed"].astype("str")

    #Random Forest 분류기를  사용한다
    rfModel_wind = RandomForestClassifier()

    # wCol에 있는 피처의 값을 바탕으로 풍속을 학습시킨다
    rfModel_wind.fit(dataWindNot0[wCol], dataWindNot0['windspeed'])

    # 학습한 값을 바탕으로 풍속이 0으로 기록된 데이터의 풍속을 예측한다
    wind0Values = rfModel_wind.predict(X = dataWind0[wCol])

    # 값을 다 예측후 비교해보기 위해 
    # 예측한 갑을 넣어 줄 데이터 프레임을 새로 만든다
    predictWind0 = dataWind0
    predictWindNot0 = dataWindNot0
    
    # 값이 0으로 기록된 풍속에 대해 예측한 값을 넣어준다
    predictWind0['windspeed'] = wind0Values
    
    # dataWindNot0 0이 아닌 풍속이 있는 데이터프레임에 예측한 값이 있는 데이터프레임을 합쳐준다
    data = predictWindNot0.append(predictWind0)
    
    # 풍속의 데이터타입을 float으로 지정해준다
    data['windspeed'] = data['windspeed'].astype('float')
    
    data.reset_index(inplace=True)
    data.drop('index', inplace=True, axis=1)
    
    return data
```


```python
 # 0값을 조정한다
train = predict_windspeed(train)
# test = predict_windspeed(test)

```


```python
fig, ax1 = plt.subplots()
fig.set_size_inches(18,6)

plt.sca(ax1)
plt.xticks(rotation=30, ha='right')
ax1.set(ylabel='count', title='train windspeed')
sns.countplot(data=train, x='windspeed', ax=ax1)
```




    <matplotlib.axes._subplots.AxesSubplot at 0xf366a58>




![png](output_12_1.png)


**Feature Selection**

- 신호와 잡음을 구분해야 한다
- 피처가 많다고 무조건 좋은 성능을 내지는 않는다
- 피처를 하나씩 추가하고 변경해 가면서 성능이 좋지않은 피처른 제거해야한다



```python
# 연속형 feature와 범주형 feature
# 연속형 feature = ['temp','humidity',windspeed','atemp']
# 범주형 feature의 type을 category로 변경해준다
categorical_feature_names = ["season", "holiday", "workingday", "weather", "dayofweek", "month", "year", "hour"]

for var in categorical_feature_names:
    train[var] = train[var].astype('category')
    test[var] = test[var].astype('category')
    

feature_names = ["season", "weather", "temp", "atemp", "humidity", "windspeed",
                 "year", "hour", "dayofweek", "holiday", "workingday"]

feature_names
```




    ['season',
     'weather',
     'temp',
     'atemp',
     'humidity',
     'windspeed',
     'year',
     'hour',
     'dayofweek',
     'holiday',
     'workingday']




```python
X_train = train[feature_names]

print(X_train.shape)
X_train.head()
```

    (10886, 11)





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }
    
    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>season</th>
      <th>weather</th>
      <th>temp</th>
      <th>atemp</th>
      <th>humidity</th>
      <th>windspeed</th>
      <th>year</th>
      <th>hour</th>
      <th>dayofweek</th>
      <th>holiday</th>
      <th>workingday</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>81</td>
      <td>0.0</td>
      <td>2011</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>9.02</td>
      <td>13.635</td>
      <td>80</td>
      <td>0.0</td>
      <td>2011</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>9.02</td>
      <td>13.635</td>
      <td>80</td>
      <td>0.0</td>
      <td>2011</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>75</td>
      <td>0.0</td>
      <td>2011</td>
      <td>3</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>75</td>
      <td>0.0</td>
      <td>2011</td>
      <td>4</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_test = test[feature_names]

print(X_test.shape)
X_test.head()
```

    (6493, 11)





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }
    
    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>season</th>
      <th>weather</th>
      <th>temp</th>
      <th>atemp</th>
      <th>humidity</th>
      <th>windspeed</th>
      <th>year</th>
      <th>hour</th>
      <th>dayofweek</th>
      <th>holiday</th>
      <th>workingday</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>11.365</td>
      <td>56</td>
      <td>26.0027</td>
      <td>2011</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>13.635</td>
      <td>56</td>
      <td>0.0000</td>
      <td>2011</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>13.635</td>
      <td>56</td>
      <td>0.0000</td>
      <td>2011</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>12.880</td>
      <td>56</td>
      <td>11.0014</td>
      <td>2011</td>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>12.880</td>
      <td>56</td>
      <td>11.0014</td>
      <td>2011</td>
      <td>4</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
label_name = "count"
y_train = train[label_name]

print(y_train.shape)
y_train.head()
```

    (10886L,)





    0    16
    1    40
    2    32
    3    13
    4     1
    Name: count, dtype: int64



**Score**

**RMSLE**

과대평가 된 항목보다는 과소평가 된 항목에 패널티를 준다.

오차(Error)를 제곱(Square)해서 평균(Mean)한 값의 제곱근(Root) 으로 값이 작을 수록 정밀도가 높다.

0에 가까운 값이 나올 수록 정밀도가 높은 값이다.

Submissions are evaluated one the Root Mean Squared Logarithmic Error (RMSLE)

$$ \sqrt{\frac{1}{n} \sum_{i=1}^n (\log(p_i + 1) - \log(a_i+1))^2 } $$
${n}$ is the number of hours in the test set
$p_i$ is your predicted count
$a_i$ is the actual count
$\log(x)$ is the natural logarithm

좀 더 자세한 설명은 : RMSLE cost function

잔차(residual)에 대한 평균에 로그를 씌운 값이다. => 과대평가 된 항목보다 과소 평가 된 항목에 패널티를 주기위해

정답에 대한 오류를 숫자로 나타낸 값으로 값이 클 수록 오차가 크다는 의미다.
값이 작을 수록 오류가 적다는 의미를 나타낸다.

![image.png](attachment:image.png)





```python
from sklearn.metrics import make_scorer

def rmsle(predicted_values, actual_values, convertExp=True):

    if convertExp:
        predicted_values = np.exp(predicted_values),
        actual_values = np.exp(actual_values)
        
    # 넘파이로 배열 형태로 바꿔준다.
    predicted_values = np.array(predicted_values)
    actual_values = np.array(actual_values)
    
    # 예측값과 실제 값에 1을 더하고 로그를 씌워준다.
    log_predict = np.log(predicted_values + 1)
    log_actual = np.log(actual_values + 1)
    
    # 위에서 계산한 예측값에서 실제값을 빼주고 제곱을 해준다.
    difference = log_predict - log_actual
    difference = np.square(difference)
    
    # 평균을 낸다.
    mean_difference = difference.mean()
    
    # 다시 루트를 씌운다.
    score = np.sqrt(mean_difference)
    
    return score
```

** 선형회귀 모델 Linear Regression Model **

- 선형회귀 또는 최소제곱법은 가장 간단하고 오래된 회귀용 선형 알고리즘
- 선형회귀는 예측과 훈련세트에 있는 타깃 y 사이의 평균제곱오차(MSE)를 최소화하는 파라미터 w와 b를 찾는다
- 매개변수가 없는 것이 장점이지만, 모델의 복잡도를 제어할 수 없다는 단점이 있다



```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import warnings
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=DeprecationWarning)

#선형회귀 모델을 초기화
lModel = LinearRegression()

# 모델을 학습시킨다
y_train_log = np.log1p(y_train)
lModel.fit(X_train,y_train_log)

# 예측하고 정확도를 평가한다
preds = lModel.predict(X_train)
print ("RMLS Value Fro linear Regression: ",
       rmsle(np.exp(y_train_log),np.exp(preds), False))
```

    ('RMLS Value Fro linear Regression: ', 0.97965334462885068)


** 릿지 Regularization Model - Ridge **

- 회귀를 위한 선형모델
- 가중치(w)의 모든 원소가 0에 가깝게 만들어 모든 피처가 주는 영향을 최소화 (기울기를 작게만듬)
- Regularization(규제)는 오버피팅(과대적합)이 되지 않도록 모델을 강제로 제한한다는 의미
- max_iter반복실행하는 최대횟수는 3000을 넣어주었다.


```python
ridge_m_ = Ridge()
ridge_params_ = { 'max_iter':[3000], 'alpha':[0.01, 0.1, 1, 2, 3, 4, 10, 30,100,200,300,400,800,900,1000]}
rmsle_scorer = metrics.make_scorer(rmsle, greater_is_better=False)
grid_ridge_m = GridSearchCV(ridge_m_,
                           ridge_params_,
                            scoring = rmsle_scorer,
                            cv=5)
y_train_log = np.log1p(y_train)
grid_ridge_m.fit( X_train, y_train_log)
preds = grid_ridge_m.predict(X_train)
print(grid_ridge_m.best_params_)
print("RMSLE Value For Ridge Regression: ", rmsle(np.exp(y_train_log),np.exp(preds), False))

fig,ax = plt.subplots()
fig.set_size_inches(12,5)
df = pd.DataFrame(grid_ridge_m.grid_scores_)
df["alpha"] = df["parameters"].apply(lambda x:x["alpha"])
df["rmsle"] = df["mean_validation_score"].apply(lambda x:-x)

plt.xticks(rotation=30, ha='right')
sns.pointplot(data=df, x='alpha',y='rmsle',ax=ax)




```

    {'alpha': 0.01, 'max_iter': 3000}
    ('RMSLE Value For Ridge Regression: ', 0.97965334261690595)





    <matplotlib.axes._subplots.AxesSubplot at 0x108c49e8>




![png](output_23_2.png)


** 라쏘 Regularization Model - Lasso **

- 선형회귀의 Regularization(규제)를 적용하는 대안
- 계수를 0에 가깝게 만들려고 하며 이를 L1규제라고 하며, 어떤 계수는 0이 되기도 하는데 이는 완전히 제외하는 피처가 생긴다는 의미다
- 피처선택이 자동으로 이루어진다고도 볼수 있다
- alpha값의 기본값은 1.0이며, 과소적합을 줄이기 위해서는 이값을 줄여야 한다
- 그리드 서치로 아래 라쏘모델을 실행했을 때 베스트 알파값은 0.0025
- max_iter(반복실행하는 최대횟수)는 3000을 넣어주었다.



```python
lasso_m_ = Lasso()

alpha  = 1/np.array([0.1, 1, 2, 3, 4, 10, 30,100,200,300,400,800,900,1000])
lasso_params_ = { 'max_iter':[3000],'alpha':alpha}

grid_lasso_m = GridSearchCV( lasso_m_,lasso_params_,scoring = rmsle_scorer,cv=5)
y_train_log = np.log1p(y_train)
grid_lasso_m.fit( X_train , y_train_log )
preds = grid_lasso_m.predict(X_train)
print (grid_lasso_m.best_params_)
print ("RMSLE Value For Lasso Regression: ",rmsle(np.exp(y_train_log),np.exp(preds),False))

fig,ax= plt.subplots()
fig.set_size_inches(12,5)
df = pd.DataFrame(grid_lasso_m.grid_scores_)
df["alpha"] = df["parameters"].apply(lambda x:x["alpha"])
df["rmsle"] = df["mean_validation_score"].apply(lambda x:-x)

plt.xticks(rotation=30, ha='right')
sns.pointplot(data=df,x="alpha",y="rmsle",ax=ax)
```

    {'alpha': 0.0025000000000000001, 'max_iter': 3000}
    ('RMSLE Value For Lasso Regression: ', 0.97966915379763486)





    <matplotlib.axes._subplots.AxesSubplot at 0x1003f048>




![png](output_25_2.png)


 **앙상블모델 - 랜덤포레스트** 

 Ensemble Models - Random Forest

 


```python
from sklearn.ensemble import RandomForestRegressor
rfModel = RandomForestRegressor(n_estimators=100)

y_train_log = np.log1p(y_train)
rfModel.fit(X_train, y_train_log)

preds = rfModel.predict(X_train)
score = rmsle(np.exp(y_train_log), np.exp(preds), False)
print("RMSLE value for Random Forest: ",score)

```

    ('RMSLE value for Random Forest: ', 0.10595027919581249)


** 앙상블모델 - 그래디언트 부스트 **

** Ensemble Model - Gradient Boost **

- 여러개의 결정트리를 묶어 강력한 모델을 만드는 또 다른 앙상블기법
- 회귀와 분류에 모두 사용할 수 있음
- 랜덤포레스트와 달리 이진트리의 오차를 보완하는 방식으로 순차적으로 트리를 만든다
- 무작위성이 없고, 강력한 사전 가지치기가 사용됨
- 1~5개의 깊지 않은 트리를 사용하기 때문에 메모리를 적게 사용하고 예측이 빠름
- learning_rate: 오차를 얼마나 강하게 보정할 것인지를 제어
- n_estimator의 값을 ㄱ키우면 앙상블에 트리가 더 많이 추가되어 모델의 복잡도가 커지고 훈련세트에서의 실수를 바로잡을 기회가 많아지지만, 너무 크면 모델이 복잡해지고 오버피팅(과적합)이 될수 있다.
- max_depth(max_leaf_nodes) 복잡도를 너무 높이지 말고 트리의 깊이가 5보다 깊어지지 않게 한다.


```python
from sklearn.ensemble import GradientBoostingRegressor
gbm = GradientBoostingRegressor(n_estimators=4000, alpha=0.01)

y_train_log = np.log1p(y_train)
gbm.fit(X_train, y_train_log)

preds = gbm.predict(X_train)
score = rmsle(np.exp(y_train_log),np.exp(preds),False)
print("RMSLE Value for Graidient Boost: ", score)
```

    ('RMSLE Value for Graidient Boost: ', 0.20417262910493975)



```python
#predsTest = rfModel.predict(X_test)
predsTest = gbm.predict(X_test)
fig, (ax1, ax2) = plt.subplots(ncols=2)
fig.set_size_inches(12,5)
sns.distplot(y_train, ax=ax1, bins=50)
sns.distplot(np.exp(predsTest), ax=ax2, bins=50)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x17ad9710>




![png](output_30_1.png)


** Submit **




```python
submission = pd.read_csv('../DATA/sampleSubmission.csv')
submission

submission["count"] = np.exp(predsTest)

print(submission.shape)
submission.head()
```

    (6493, 2)





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }
    
    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011-01-20 00:00:00</td>
      <td>12.449856</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-01-20 01:00:00</td>
      <td>5.658840</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-01-20 02:00:00</td>
      <td>3.825400</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-01-20 03:00:00</td>
      <td>2.586785</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011-01-20 04:00:00</td>
      <td>2.281436</td>
    </tr>
  </tbody>
</table>
</div>




```python
submission.to_csv('../DATA/Score_{0:.5f}_submission.csv'.format(score),index=False)
```


```python

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
```


```python
from sklearn.ensemble import RandomForestRegressor

max_depth_list = []

model = RandomForestRegressor(n_estimators=100,
                              n_jobs=-1,
                              random_state=0)
model
```




    RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
               max_features='auto', max_leaf_nodes=None,
               min_impurity_decrease=0.0, min_impurity_split=None,
               min_samples_leaf=1, min_samples_split=2,
               min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=-1,
               oob_score=False, random_state=0, verbose=0, warm_start=False)




```python
%time score = cross_val_score(model, X_train, y_train, cv=k_fold, scoring=rmsle_scorer)
score = score.mean()
# 0에 근접할수록 좋은 데이터
print("Score= {0:.5f}".format(score))
```

    Wall time: 17.9 s
    Score= 0.33063



```python
# 학습시킴, 피팅(옷을 맞출 때 사용하는 피팅을 생각함) - 피처와 레이블을 넣어주면 알아서 학습을 함
model.fit(X_train, y_train)
```




    RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
               max_features='auto', max_leaf_nodes=None,
               min_impurity_decrease=0.0, min_impurity_split=None,
               min_samples_leaf=1, min_samples_split=2,
               min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=-1,
               oob_score=False, random_state=0, verbose=0, warm_start=False)




```python

# 예측
predictions = model.predict(X_test)

print(predictions.shape)
predictions[0:10]
```

    (6493L,)





    array([  12.68,    5.02,    4.11,    3.6 ,    3.19,    6.33,   38.23,
            105.43,  233.31,  136.18])




```python

# 예측한 데이터를 시각화 해본다. 
fig,(ax1,ax2)= plt.subplots(ncols=2)
fig.set_size_inches(12,5)
sns.distplot(y_train,ax=ax1,bins=50)
ax1.set(title="train")
sns.distplot(predictions,ax=ax2,bins=50)
ax2.set(title="test")
```




    [<matplotlib.text.Text at 0x14e2cd30>]




![png](output_39_1.png)



```python
submission = pd.read_csv("../DATA/sampleSubmission.csv")
submission

submission["count"] = predictions

print(submission.shape)
submission.head()
```

    (6493, 2)





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }
    
    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011-01-20 00:00:00</td>
      <td>12.68</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-01-20 01:00:00</td>
      <td>5.02</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-01-20 02:00:00</td>
      <td>4.11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-01-20 03:00:00</td>
      <td>3.60</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011-01-20 04:00:00</td>
      <td>3.19</td>
    </tr>
  </tbody>
</table>
</div>




```python
submission.to_csv("../DATA/Score_{0:.5f}_submission.csv".format(score), index=False)
```
