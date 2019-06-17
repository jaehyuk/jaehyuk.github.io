```
---
layout: post
title:  "Statoil-modeling"
date:   2019-06167
categories: Statoil
---
```



data preprocessing : https://www.kaggle.com/muonneutrino/exploration-transforming-images-in-python

keras CNN: https://www.kaggle.com/devm2024/keras-model-for-beginners-0-210-on-lb-eda-r-d

HIstory: https://www.kaggle.com/cbryant/keras-cnn-statoil-iceberg-lb-0-1995-now-0-1516


```python
from google.colab import files
files.upload()

```



     <input type="file" id="files-7ca3b656-0bb1-43bf-8bb1-7974be7db6ee" name="files[]" multiple disabled />
     <output id="result-7ca3b656-0bb1-43bf-8bb1-7974be7db6ee">
      Upload widget is only available when the cell has been executed in the
      current browser session. Please rerun this cell to enable.
      </output>
      <script src="/nbextensions/google.colab/files.js"></script> 


    Saving kaggle.json to kaggle.json





    {'kaggle.json': b'{"username":"hohobrothers","key":"2c60c5affdffcdb12d3474eebf961abf"}'}




```python
# Next, install kaggle API client
#!pip install -q kaggle

# The Kaggle API client expects this file to be in ~/.kaggle,
# so move it there.
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/

# This permissions change avoids a warning on Kaggle tool startup.
!chmod 600 ~/.kaggle/kaggle.json

# Kaggle dataset download
# copy from Kaggle dataset website
```


```python
!kaggle competitions download -c statoil-iceberg-classifier-challenge
```

    Downloading sample_submission.csv.7z to /content
      0% 0.00/37.7k [00:00<?, ?B/s]
    100% 37.7k/37.7k [00:00<00:00, 32.1MB/s]
    Downloading train.json.7z to /content
     96% 41.0M/42.9M [00:01<00:00, 43.0MB/s]
    100% 42.9M/42.9M [00:01<00:00, 42.1MB/s]
    Downloading test.json.7z to /content
     95% 233M/245M [00:02<00:00, 86.8MB/s]
    100% 245M/245M [00:02<00:00, 104MB/s] 



```python
!7z x train.json.7z
!7z x test.json.7z
```


    7-Zip [64] 16.02 : Copyright (c) 1999-2016 Igor Pavlov : 2016-05-21
    p7zip Version 16.02 (locale=en_US.UTF-8,Utf16=on,HugeFiles=on,64 bits,2 CPUs Intel(R) Xeon(R) CPU @ 2.30GHz (306F0),ASM,AES-NI)
    
    Scanning the drive for archives:
      0M Scan         1 file, 44932785 bytes (43 MiB)
    
    Extracting archive: train.json.7z
    --
    Path = train.json.7z
    Type = 7z
    Physical Size = 44932785
    Headers Size = 154
    Method = LZMA2:24
    Solid = -
    Blocks = 1
    
      0%      6% - data/processed/train.json                                 12% - data/processed/train.json                                 19% - data/processed/train.json                                 25% - data/processed/train.json                                 32% - data/processed/train.json                                 38% - data/processed/train.json                                 44% - data/processed/train.json                                 51% - data/processed/train.json                                 57% - data/processed/train.json                                 64% - data/processed/train.json                                 70% - data/processed/train.json                                 76% - data/processed/train.json                                 83% - data/processed/train.json                                 88% - data/processed/train.json                                 95% - data/processed/train.json                                Everything is Ok
    
    Size:       196313674
    Compressed: 44932785
    
    7-Zip [64] 16.02 : Copyright (c) 1999-2016 Igor Pavlov : 2016-05-21
    p7zip Version 16.02 (locale=en_US.UTF-8,Utf16=on,HugeFiles=on,64 bits,2 CPUs Intel(R) Xeon(R) CPU @ 2.30GHz (306F0),ASM,AES-NI)
    
    Scanning the drive for archives:
      0M Scan         1 file, 257127394 bytes (246 MiB)
    
    Extracting archive: test.json.7z
    --
    Path = test.json.7z
    Type = 7z
    Physical Size = 257127394
    Headers Size = 154
    Method = LZMA2:24
    Solid = -
    Blocks = 1
    
      0%      1% - data/processed/test.json                                 2% - data/processed/test.json                                 3% - data/processed/test.json                                 4% - data/processed/test.json                                 5% - data/processed/test.json                                 6% - data/processed/test.json                                 8% - data/processed/test.json                                 9% - data/processed/test.json                                10% - data/processed/test.json                                11% - data/processed/test.json                                12% - data/processed/test.json                                13% - data/processed/test.json                                15% - data/processed/test.json                                16% - data/processed/test.json                                17% - data/processed/test.json                                18% - data/processed/test.json                                19% - data/processed/test.json                                20% - data/processed/test.json                                21% - data/processed/test.json                                22% - data/processed/test.json                                24% - data/processed/test.json                                25% - data/processed/test.json                                26% - data/processed/test.json                                27% - data/processed/test.json                                28% - data/processed/test.json                                29% - data/processed/test.json                                30% - data/processed/test.json                                32% - data/processed/test.json                                33% - data/processed/test.json                                34% - data/processed/test.json                                35% - data/processed/test.json                                36% - data/processed/test.json                                38% - data/processed/test.json                                39% - data/processed/test.json                                40% - data/processed/test.json                                41% - data/processed/test.json                                42% - data/processed/test.json                                43% - data/processed/test.json                                44% - data/processed/test.json                                46% - data/processed/test.json                                47% - data/processed/test.json                                48% - data/processed/test.json                                49% - data/processed/test.json                                50% - data/processed/test.json                                51% - data/processed/test.json                                52% - data/processed/test.json                                54% - data/processed/test.json                                55% - data/processed/test.json                                56% - data/processed/test.json                                57% - data/processed/test.json                                58% - data/processed/test.json                                60% - data/processed/test.json                                61% - data/processed/test.json                                62% - data/processed/test.json                                63% - data/processed/test.json                                64% - data/processed/test.json                                65% - data/processed/test.json                                66% - data/processed/test.json                                68% - data/processed/test.json                                69% - data/processed/test.json                                70% - data/processed/test.json                                71% - data/processed/test.json                                72% - data/processed/test.json                                73% - data/processed/test.json                                74% - data/processed/test.json                                76% - data/processed/test.json                                77% - data/processed/test.json                                78% - data/processed/test.json                                79% - data/processed/test.json                                80% - data/processed/test.json                                81% - data/processed/test.json                                82% - data/processed/test.json                                84% - data/processed/test.json                                85% - data/processed/test.json                                86% - data/processed/test.json                                87% - data/processed/test.json                                88% - data/processed/test.json                                90% - data/processed/test.json                                91% - data/processed/test.json                                92% - data/processed/test.json                                93% - data/processed/test.json                                94% - data/processed/test.json                                95% - data/processed/test.json                                97% - data/processed/test.json                                98% - data/processed/test.json                                99% - data/processed/test.json                               Everything is Ok
    
    Size:       1521771850
    Compressed: 257127394



```python
#pip install pyunpack
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from os.path import join as opj
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pylab
plt.rcParams['figure.figsize'] = 10, 10
%matplotlib inline
```


```python
train = pd.read_json("./data/processed/train.json", encoding='utf8')
test = pd.read_json("./data/processed/test.json", encoding='utf8')
#train = pd.read_json("./input/data/processed/train.json", encoding='utf8')
#test = pd.read_json("./input/data/processed/test.json", encoding='utf8')
```


```python
import missingno as msno
msno.matrix(train)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f79ff06b4a8>




![png](output_7_1.png)



```python
train.describe().T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>is_iceberg</th>
      <td>1604.0</td>
      <td>0.469451</td>
      <td>0.499222</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_stats = train.drop(['id', 'is_iceberg', 'band_1','band_2'], axis=1)
```


```python
from scipy import signal

# smoothing filter
smooth = np.array([[1,1,1],[1,5,1],[1,1,1]])

# 1st derivative
xder = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
yder = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])

xder2 = np.array([[-1,2,-1],[-3,6,-3],[-1,2,-1]])
yder2 = np.array([[-1,-3,-1],[2,6,2],[-1,-3,-1]])
```


```python
train.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>band_1</th>
      <th>band_2</th>
      <th>id</th>
      <th>inc_angle</th>
      <th>is_iceberg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[-27.878360999999998, -27.15416, -28.668615, -...</td>
      <td>[-27.154118, -29.537888, -31.0306, -32.190483,...</td>
      <td>dfd5f913</td>
      <td>43.9239</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[-12.242375, -14.920304999999999, -14.920363, ...</td>
      <td>[-31.506321, -27.984554, -26.645678, -23.76760...</td>
      <td>e25388fd</td>
      <td>38.1562</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[-24.603676, -24.603714, -24.871029, -23.15277...</td>
      <td>[-24.870956, -24.092632, -20.653963, -19.41104...</td>
      <td>58b2aaa0</td>
      <td>45.2859</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[-22.454607, -23.082819, -23.998013, -23.99805...</td>
      <td>[-27.889421, -27.519794, -27.165262, -29.10350...</td>
      <td>4cfc3a18</td>
      <td>43.8306</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[-26.006956, -23.164886, -23.164886, -26.89116...</td>
      <td>[-27.206915, -30.259186, -30.259186, -23.16495...</td>
      <td>271f93f4</td>
      <td>35.6256</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
train.loc[train.is_iceberg==1]['band_1']#.iloc[0]`
```




    2       [-24.603676, -24.603714, -24.871029, -23.15277...
    5       [-20.769371, -20.769434, -25.906025, -25.90602...
    6       [-26.673811, -23.666162, -27.622442, -28.31768...
    10      [-21.397552, -19.753859, -23.426783, -24.65221...
    12      [-21.112206, -21.638832, -25.436468, -23.22255...
    13      [-23.864258, -27.755791, -26.047226, -24.62014...
    19      [-21.557806, -22.084446, -19.187838, -16.91901...
    21      [-26.345797, -26.05139, -26.650714999999998, -...
    23      [-14.6148, -14.6148, -16.136662, -15.342532, -...
    25      [-19.261448, -19.671938, -20.712574, -20.10284...
    26      [-10.99748, -11.994458, -11.209444, -11.209444...
    28      [-27.995171, -23.865738, -22.165567, -22.76487...
    32      [-34.254326, -29.97677, -28.233807, -27.123505...
    33      [-24.426573, -22.591206, -23.46056, -24.685993...
    34      [-19.418335, -19.166628, -19.043499, -20.50571...
    35      [-17.934721, -18.271059, -19.900589, -19.11048...
    36      [-20.464077, -23.416199, -23.625576, -20.31829...
    38      [-25.560505, -25.560505, -27.303579, -22.67576...
    39      [-25.204479, -22.638159, -21.328127, -21.68275...
    40      [-23.890957, -21.813034, -21.999882, -18.99681...
    46      [-16.603346, -19.007256, -17.298717, -19.5261,...
    47      [-25.822117, -24.994314, -25.261593, -26.11668...
    49      [-20.833214, -21.799376, -23.485867, -25.84792...
    50      [-26.338024, -26.338083, -23.990429, -21.78223...
    51      [-24.888716, -24.888771, -24.888828, -23.47726...
    53      [-18.96467, -21.967827, -21.071625, -20.737555...
    54      [-22.161348, -20.517677, -20.369263, -18.7612,...
    58      [-22.426252, -19.744678, -18.026453, -21.88901...
    60      [-18.2882, -23.949383, -24.187414, -21.688688,...
    61      [-24.389923, -22.769339, -22.407455, -20.79441...
                                  ...                        
    1452    [-18.52878, -17.601404, -18.287546, -22.112103...
    1457    [-28.143589, -23.556904, -22.913263, -24.74872...
    1459    [-28.201336, -24.420252, -21.425047, -21.07051...
    1460    [-24.066139, -22.665459, -23.851814, -24.51181...
    1471    [-22.518232, -21.945328, -26.139376, -28.15283...
    1472    [-25.277325, -26.753099, -26.753149, -24.75089...
    1473    [-32.045311, -30.46174, -28.332691, -32.045479...
    1474    [-25.749205, -22.956005, -20.189989, -21.01787...
    1475    [-20.591736, -19.982, -20.435331, -24.769514, ...
    1476    [-24.891918, -23.17366, -25.452568, -26.367756...
    1477    [-21.781345, -22.521074, -23.329704, -20.62167...
    1478    [-20.712835, -21.943243, -23.157244, -24.82820...
    1479    [-18.77581, -16.39217, -18.222559, -18.11598, ...
    1481    [-21.264984, -21.813789, -23.027786, -22.00477...
    1484    [-11.033886, -11.033886, -14.7217, -14.50906, ...
    1486    [-27.532944, -24.610439, -25.388853, -28.25735...
    1487    [-18.151741, -16.310638, -17.797262, -22.62916...
    1489    [-25.649862, -23.596659, -24.562798, -25.08942...
    1490    [-28.360956, -28.747118, -28.747118, -26.96835...
    1492    [-24.61931, -18.598782, -18.598782, -18.141596...
    1493    [-17.487677, -17.383654, -14.000873, -15.78294...
    1496    [-19.772322, -20.066835, -18.433485, -20.06693...
    1497    [-19.516977, -16.813873, -11.867026, -12.89722...
    1498    [-23.797707, -23.797758, -24.763905, -25.56629...
    1500    [-19.91123, -16.81319, -16.343637, -17.516617,...
    1501    [-23.322561, -23.122875, -23.322632, -22.18815...
    1502    [-16.801426, -14.727657, -16.608463, -16.99897...
    1504    [-20.528059, -22.144562, -26.714188, -24.64341...
    1506    [-23.945009, -23.316914, -25.883354, -22.36160...
    1508    [-16.241388, -17.401228, -18.385656, -19.23677...
    Name: band_1, Length: 753, dtype: object




```python
# Plot band_1
fig = plt.figure(1,figsize=(15,15))
for i in range(9):
    ax = fig.add_subplot(3,3,i+1)
    arr = signal.convolve2d(np.reshape(np.array(train.loc[train.is_iceberg==1]),(75,75)),xder,boundary='fill',mode='same')
    ax.imshow(arr,cmap='inferno')
    ax.set_title('iceberg: X-derivative')
    
plt.show()
```


    ---------------------------------------------------------------------------
    
    ValueError                                Traceback (most recent call last)
    
    <ipython-input-17-fc29b433f854> in <module>()
          2 for i in range(9):
          3     ax = fig.add_subplot(3,3,i+1)
    ----> 4     arr = signal.convolve2d(np.reshape(np.array(train.loc[train.is_iceberg==1]),(75,75)),xder,boundary='fill',mode='same')
          5     ax.imshow(arr,cmap='inferno')
          6     ax.set_title('iceberg: X-derivative')


    /usr/local/lib/python3.6/dist-packages/numpy/core/fromnumeric.py in reshape(a, newshape, order)
        290            [5, 6]])
        291     """
    --> 292     return _wrapfunc(a, 'reshape', newshape, order=order)
        293 
        294 


    /usr/local/lib/python3.6/dist-packages/numpy/core/fromnumeric.py in _wrapfunc(obj, method, *args, **kwds)
         54 def _wrapfunc(obj, method, *args, **kwds):
         55     try:
    ---> 56         return getattr(obj, method)(*args, **kwds)
         57 
         58     # An AttributeError occurs if the object does not have


    ValueError: cannot reshape array of size 3765 into shape (75,75)



![png](output_13_1.png)



```python
band_1_2d = np.array([np.array(band).astype(np.float32).reshape(75,75) for band in train['band_1']])
plt.imshow(signal.convolve2d(band_1_2d[0],xder,mode='valid'))

#band_2_1st = np.array([np.array(band).astype(np.float32).reshape(75,75) for band in train['band_2']])




```




    <matplotlib.image.AxesImage at 0x7f79ac6e93c8>




![png](output_14_1.png)



```python
band_1_2d = np.array([np.array(band).astype(np.float32).reshape(75,75) for band in train['band_1']])
```


```python
band_1_1st_der = []
for i in range(band_1_2d.shape[0]):
    band_1_1st_der.append(signal.convolve2d(band_1_2d[i],xder,boundary='fill', mode='same') )
```


```python
len(band_1_1st_der), band_1_1st_der[100].shape
```




    (1604, (75, 75))




```python
print(np.newaxis)
```

    None



```python
band_1 = np.array([np.array(band).astype(np.float32).reshape(75,75) for band in train['band_1']])
band_2 = np.array([np.array(band).astype(np.float32).reshape(75,75) for band in train['band_2']])
```


```python
band_1_der = np.array([signal.convolve2d(np.array(band).astype(np.float32).reshape(75,75),xder, mode='same') for band in train['band_1']])
band_2_der = np.array([signal.convolve2d(np.array(band).astype(np.float32).reshape(75,75),xder, mode='same') for band in train['band_2']])

train_all = np.concatenate([band_1[:,:,:,np.newaxis], band_2[:,:,:,np.newaxis],  
                            band_1_der[:,:,:,np.newaxis], band_2_der[:,:,:,np.newaxis]], axis=-1)

```


```python
print("band_1 shape: ", band_1.shape)
print("band_1_der shape: ", band_1_der.shape)
```

    band_1 shape:  (1604, 75, 75)
    band_1_der shape:  (1604, 75, 75)



```python
#band_1_der = np.array([signal.convolve2d(np.array(band).astype(np.float32).reshape(75,75),xder,boundary='fill', mode='valid') for band in train['band_1']])
#band_2_der = np.array([signal.convolve2d(np.array(band).astype(np.float32).reshape(75,75),xder,boundary='fill', mode='valid') for band in train['band_2']])

```


```python
train_all.shape
```




    (1604, 75, 75, 4)




```python
band_1_test = np.array([np.array(band).astype(np.float32).reshape(75,75) for band in test['band_1']])
band_2_test = np.array([np.array(band).astype(np.float32).reshape(75,75) for band in test['band_2']])
```


```python
band_1_test_der = np.array([signal.convolve2d(np.array(band).astype(np.float32).reshape(75,75),xder, mode='same') for band in test['band_1']])
band_2_test_der = np.array([signal.convolve2d(np.array(band).astype(np.float32).reshape(75,75),xder, mode='same') for band in test['band_2']])

test_all = np.concatenate([band_1_test[:,:,:,np.newaxis], band_2_test[:,:,:,np.newaxis],  
                            band_1_test_der[:,:,:,np.newaxis], band_2_test_der[:,:,:,np.newaxis]], axis=-1)

```


```python
# Cross_validation
# # https://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/
# # define 10-fold cross validation test harness
# kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019)
# target = train['is_iceberg']
# cvscores = []
# for train, valid in kfold.split(train_all, target):
#   # create model
# 	model = Sequential()
# 	model.add(Dense(12, input_dim=8, activation='relu'))
# 	model.add(Dense(8, activation='relu'))
# 	model.add(Dense(1, activation='sigmoid'))
# 	# Compile model
# 	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# 	# Fit the model
# 	model.fit(X[train], Y[train], epochs=150, batch_size=10, verbose=0)
# 	# evaluate the model
# 	scores = model.evaluate(X[test], Y[test], verbose=0)
# 	print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# 	cvscores.append(scores[1] * 100)
# print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
```


```python
num = 100
label = 'iceberg' if (train['is_iceberg'].values[num]==1) else 'ship'
plot_contour_2d(X_band_1[num,:,:], X_band_2[num,:,:], label)
```


```python

```

# New Section


```python
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt

# #plt.rcParams['figure, figsize'] = 12, 8
# %matplotlib inline

# # Take a look at a iceberg
# import plotly.offline as py
# import plotly.graph_objs as go
# from plotly import tools

# py.init_notebook_mode(connected=True) # plotly를 jupyter notebook에 사용

# # tqdm
# #from tqdm import tqdm_notebook, tnrange
```


<script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script><script type="text/javascript">if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script><script>requirejs.config({paths: { 'plotly': ['https://cdn.plot.ly/plotly-latest.min']},});if(!window._Plotly) {require(['plotly'],function(plotly) {window._Plotly=plotly;});}</script>



```python
import os
import zipfile
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# #local_zip = '/tmp/train.json.7z'
# #Archive(local_zip).extractall('/tmp')


# # zip_ref = zipfile.ZipFile(local_zip, 'r')
# # zip_ref.extractall('/tmp')
# # zip_ref.close()

# base_dir = '/content/'
# train_dir = os.path.join(base_dir, 'train.json')
# test_dir = os.path.join(base_dir, 'test.json')
```


```python
# train = pd.read_json('./input/data/processed/train.json')
# print('train data done!')

```

http://deliverableinsights.com/index.php/2018/02/04/iceberg-challenge/

https://github.com/bmowry06/IcebergChallenge

###  What is log loss?
### Introduction
Log Loss is the most important classification metric based on probabilities. 

It's hard to interpret raw log-loss values, but log-loss is still a good metric for comparing models.  For any given problem, a lower log-loss value means better predictions.

Log Loss is a slight twist on something called the **Likelihood Function**. In fact, Log Loss is -1 * the log of the likelihood function. So, we will start by understanding the likelihood function.

The likelihood function answers the question "How likely did the model think the actually observed set of outcomes was." If that sounds confusing, an example should help.  

### Example
A model predicts probabilities of `[0.8, 0.4, 0.1]` for three houses.  The first two houses were sold, and the last one was not sold. So the actual outcomes could be represented numeically as `[1, 1, 0]`.

Let's step through these predictions one at a time to iteratively calculate the likelihood function.

The first house sold, and the model said that was 80% likely.  So, the likelihood function after looking at one prediction is 0.8.

The second house sold, and the model said that was 40% likely.  There is a rule of probability that the probability of multiple independent events is the product of their individual probabilities.  So, we get the combined likelihood from the first two predictions by multiplying their associated probabilities.  That is `0.8 * 0.4`, which happens to be 0.32.

Now we get to our third prediction.  That home did not sell.  The model said it was 10% likely to sell.  That means it was 90% likely to not sell.  So, the observed outcome of *not selling* was 90% likely according to the model.  So, we multiply the previous result of 0.32 by 0.9.  

We could step through all of our predictions.  Each time we'd find the probability associated with the outcome that actually occurred, and we'd multiply that by the previous result.  That's the likelihood.

### From Likelihood to Log Loss
Each prediction is between 0 and 1. If you multiply enough numbers in this range, the result gets so small that computers can't keep track of it.  So, as a clever computational trick, we instead keep track of the log of the Likelihood.  This is in a range that's easy to keep track of. We multiply this by negative 1 to maintain a common convention that lower loss scores are better.



https://datawookie.netlify.com/blog/2015/12/making-sense-of-logarithmic-loss/




```
import os
import numpy as np 
import pandas as pd 
from sklearn.metrics import log_loss 
%pylab inline
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
```

---







```python
train.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>band_1</th>
      <th>band_2</th>
      <th>id</th>
      <th>inc_angle</th>
      <th>is_iceberg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[-27.878360999999998, -27.15416, -28.668615, -...</td>
      <td>[-27.154118, -29.537888, -31.0306, -32.190483,...</td>
      <td>dfd5f913</td>
      <td>43.9239</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[-12.242375, -14.920304999999999, -14.920363, ...</td>
      <td>[-31.506321, -27.984554, -26.645678, -23.76760...</td>
      <td>e25388fd</td>
      <td>38.1562</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[-24.603676, -24.603714, -24.871029, -23.15277...</td>
      <td>[-24.870956, -24.092632, -20.653963, -19.41104...</td>
      <td>58b2aaa0</td>
      <td>45.2859</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[-22.454607, -23.082819, -23.998013, -23.99805...</td>
      <td>[-27.889421, -27.519794, -27.165262, -29.10350...</td>
      <td>4cfc3a18</td>
      <td>43.8306</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[-26.006956, -23.164886, -23.164886, -26.89116...</td>
      <td>[-27.206915, -30.259186, -30.259186, -23.16495...</td>
      <td>271f93f4</td>
      <td>35.6256</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
band_1 = np.array([np.array(band).astype(np.float32).reshape(75,75) for band in train['band_1']])
band_2 = np.array([np.array(band).astype(np.float32).reshape(75,75) for band in train['band_2']])

#train_all = np.concatenate([band_1[:,:,:,np.newaxis], band_2[:,:,:,np.newaxis], ((band_1 + band_2)/2)[:,:,:, np.newaxis]], axis=-1)
train_all = np.concatenate([band_1[:,:,:,np.newaxis], band_2[:,:,:,np.newaxis]], axis=-1)
```


```python
target = train['is_iceberg']
```


```python
train_cv, train_valid, target_cv, target_valid = train_test_split(train_all, target, random_state=2019, train_size=0.8)
```


```python

```

### Building CNN model using Keras. 


```python
from matplotlib import pyplot
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop, Adam
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, Activation
from keras.layers import GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate

from keras.models import Model
from keras import initializers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
```

    Using TensorFlow backend.



```python

```


```python
# class myCallback(tf.keras.callbacks.Callback):
#     def on_epochs_end(self, epoch, logs={}):
#         if (logs.get('acc')>0.99):
#             print('stop modeling')
#             self.model.stop_training = True
          
            
```


```python
def get_callbacks(filepath, patience=2):
    early_stopping = EarlyStopping('val_loss', patience=patience, mode='min')
    msave = ModelCheckpoint(filepath, save_best_only=True)
    return [early_stopping, msave]

file_path = '/content/model_weights.hdf5'
callbacks = get_callbacks(file_path, patience=5) 
```


```python
#callback = myCallback()
```


```python
import tensorflow as tf
model = tf.keras.models.Sequential([
    # 1st layer
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(75, 75, 2)),
    tf.keras.layers.MaxPooling2D(3,2),
    tf.keras.layers.Dropout(0.2),
    # 2nd layer
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.2),
    # 3rd layer
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.2),
    # 4th layer
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.2),
    # Flatten
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])
```


```python
model.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_4 (Conv2D)            (None, 73, 73, 64)        1216      
    _________________________________________________________________
    max_pooling2d_4 (MaxPooling2 (None, 36, 36, 64)        0         
    _________________________________________________________________
    dropout_4 (Dropout)          (None, 36, 36, 64)        0         
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 34, 34, 128)       73856     
    _________________________________________________________________
    max_pooling2d_5 (MaxPooling2 (None, 17, 17, 128)       0         
    _________________________________________________________________
    dropout_5 (Dropout)          (None, 17, 17, 128)       0         
    _________________________________________________________________
    conv2d_6 (Conv2D)            (None, 15, 15, 128)       147584    
    _________________________________________________________________
    max_pooling2d_6 (MaxPooling2 (None, 7, 7, 128)         0         
    _________________________________________________________________
    dropout_6 (Dropout)          (None, 7, 7, 128)         0         
    _________________________________________________________________
    conv2d_7 (Conv2D)            (None, 5, 5, 64)          73792     
    _________________________________________________________________
    max_pooling2d_7 (MaxPooling2 (None, 2, 2, 64)          0         
    _________________________________________________________________
    dropout_7 (Dropout)          (None, 2, 2, 64)          0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 256)               0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 512)               131584    
    _________________________________________________________________
    dense_4 (Dense)              (None, 256)               131328    
    _________________________________________________________________
    dense_5 (Dense)              (None, 1)                 257       
    =================================================================
    Total params: 559,617
    Trainable params: 559,617
    Non-trainable params: 0
    _________________________________________________________________



```python
#model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])
model.compile(optimizer="Adam", loss='binary_crossentropy', metrics=['acc'])
```


```python
history = model.fit(train_cv, target_cv,
                   batch_size=24,
                   epochs=50, 
                   verbose=1,
                   validation_data=(train_valid, target_valid),
                   callbacks=callbacks
                   )
```

    Train on 1283 samples, validate on 321 samples
    Epoch 1/50
    1283/1283 [==============================] - 1s 988us/sample - loss: 1.0605 - acc: 0.5651 - val_loss: 0.5845 - val_acc: 0.6604
    Epoch 2/50
    1283/1283 [==============================] - 1s 573us/sample - loss: 0.5649 - acc: 0.6648 - val_loss: 0.5443 - val_acc: 0.7009
    Epoch 3/50
    1283/1283 [==============================] - 1s 580us/sample - loss: 0.5205 - acc: 0.7194 - val_loss: 0.5017 - val_acc: 0.7850
    Epoch 4/50
    1283/1283 [==============================] - 1s 568us/sample - loss: 0.4849 - acc: 0.7514 - val_loss: 0.4339 - val_acc: 0.7726
    Epoch 5/50
    1283/1283 [==============================] - 1s 567us/sample - loss: 0.4384 - acc: 0.7786 - val_loss: 0.4256 - val_acc: 0.8100
    Epoch 6/50
    1283/1283 [==============================] - 1s 567us/sample - loss: 0.4123 - acc: 0.8028 - val_loss: 0.3688 - val_acc: 0.8100
    Epoch 7/50
    1283/1283 [==============================] - 1s 574us/sample - loss: 0.3816 - acc: 0.8246 - val_loss: 0.3876 - val_acc: 0.8224
    Epoch 8/50
    1283/1283 [==============================] - 1s 567us/sample - loss: 0.3578 - acc: 0.8324 - val_loss: 0.3426 - val_acc: 0.8287
    Epoch 9/50
    1283/1283 [==============================] - 1s 569us/sample - loss: 0.3497 - acc: 0.8465 - val_loss: 0.3414 - val_acc: 0.8442
    Epoch 10/50
    1283/1283 [==============================] - 1s 563us/sample - loss: 0.3482 - acc: 0.8433 - val_loss: 0.3268 - val_acc: 0.8692
    Epoch 11/50
    1283/1283 [==============================] - 1s 569us/sample - loss: 0.3353 - acc: 0.8402 - val_loss: 0.3124 - val_acc: 0.8536
    Epoch 12/50
    1283/1283 [==============================] - 1s 782us/sample - loss: 0.3206 - acc: 0.8535 - val_loss: 0.2634 - val_acc: 0.8847
    Epoch 13/50
    1283/1283 [==============================] - 1s 562us/sample - loss: 0.3438 - acc: 0.8441 - val_loss: 0.3112 - val_acc: 0.8723
    Epoch 14/50
    1283/1283 [==============================] - 1s 571us/sample - loss: 0.3341 - acc: 0.8402 - val_loss: 0.2766 - val_acc: 0.8660
    Epoch 15/50
    1283/1283 [==============================] - 1s 573us/sample - loss: 0.2828 - acc: 0.8605 - val_loss: 0.3045 - val_acc: 0.8505
    Epoch 16/50
    1283/1283 [==============================] - 1s 575us/sample - loss: 0.3011 - acc: 0.8644 - val_loss: 0.3644 - val_acc: 0.8380
    Epoch 17/50
    1283/1283 [==============================] - 1s 569us/sample - loss: 0.2918 - acc: 0.8535 - val_loss: 0.2916 - val_acc: 0.8567





```python
%matplotlib inline

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', 'Validation Accuracy')
plt.figure()

plt.plot(epochs, loss, 'r', 'Training Loss')
plt.plot(epochs, val_loss, 'b', 'Validation Accuracy')
plt.figure()
```




    <Figure size 432x288 with 0 Axes>




![png](output_53_1.png)



![png](output_53_2.png)



    <Figure size 432x288 with 0 Axes>



```python
score = model.evaluate(train_valid, target_valid, verbose=1)
print('Test Loss:', score[0])
print('Test Accuracy:', score[1])
```

    321/321 [==============================] - 0s 333us/sample - loss: 0.2916 - acc: 0.8567
    Test Loss: 0.29164910724972637
    Test Accuracy: 0.8566978



```python
test_band_1 = np.array([np.array(test_band).astype(np.float32).reshape(75,75) for test_band in test['band_1']])
test_band_2 = np.array([np.array(test_band).astype(np.float32).reshape(75,75) for test_band in test['band_2']])

#test_all = np.concatenate([test_band_1[:,:,:,np.newaxis], test_band_2[:,:,:,np.newaxis], ((test_band_1 + test_band_2)/2)[:,:,:, np.newaxis]], axis=-1)
test_all = np.concatenate([test_band_1[:,:,:,np.newaxis], test_band_2[:,:,:,np.newaxis]], axis=-1)
```


```python
# Prediction
predicted_test=model.predict_proba(test_all)
```




```python
import datetime
now = datetime.datetime.now()

```

### Submission


```python
time_stamp = now.strftime("%Y%m%d_%H%M%S")
submission = pd.DataFrame()
submission['id'] = test['id']
submission['is_iceberg']=predicted_test.reshape((predicted_test.shape[0]))
filename = '7th_submission_{0:s}.csv'.format(time_stamp)
submission.to_csv(filename, index=False)
```


```python
submission.head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>is_iceberg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5941774d</td>
      <td>0.229351</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4023181e</td>
      <td>0.547611</td>
    </tr>
    <tr>
      <th>2</th>
      <td>b20200e4</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>e7f018bb</td>
      <td>0.998192</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4371c8c3</td>
      <td>0.495040</td>
    </tr>
    <tr>
      <th>5</th>
      <td>a8d9b1fd</td>
      <td>0.047891</td>
    </tr>
    <tr>
      <th>6</th>
      <td>29e7727e</td>
      <td>0.094357</td>
    </tr>
    <tr>
      <th>7</th>
      <td>92a51ffb</td>
      <td>0.996090</td>
    </tr>
    <tr>
      <th>8</th>
      <td>c769ac97</td>
      <td>0.000013</td>
    </tr>
    <tr>
      <th>9</th>
      <td>aee0547d</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>565b28ac</td>
      <td>0.000015</td>
    </tr>
    <tr>
      <th>11</th>
      <td>e04e9775</td>
      <td>0.575174</td>
    </tr>
    <tr>
      <th>12</th>
      <td>8e8161d1</td>
      <td>0.016982</td>
    </tr>
    <tr>
      <th>13</th>
      <td>4cf4d256</td>
      <td>0.316664</td>
    </tr>
    <tr>
      <th>14</th>
      <td>139e5324</td>
      <td>0.000251</td>
    </tr>
    <tr>
      <th>15</th>
      <td>f156976f</td>
      <td>0.003821</td>
    </tr>
    <tr>
      <th>16</th>
      <td>68a117cc</td>
      <td>0.000238</td>
    </tr>
    <tr>
      <th>17</th>
      <td>d9aa7a56</td>
      <td>0.001462</td>
    </tr>
    <tr>
      <th>18</th>
      <td>9005b143</td>
      <td>0.791574</td>
    </tr>
    <tr>
      <th>19</th>
      <td>5f6d3988</td>
      <td>0.950368</td>
    </tr>
  </tbody>
</table>
</div>




```python
predicted_test.shape
```




    (8424, 1)




```python

```