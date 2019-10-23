---
layout: splash
title: "Kaggle: Detect diabetic retinopathy to stop blindness before it's too late"
date:   2019-10-22 11:13:00 -0500
---

### Kaggle: APTOS 2019 Blindness Detection 


```python
#https://github.com/keras-team/keras/issues/4161#issuecomment-366031228
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras
```

    Using TensorFlow backend.
    


```python
import matplotlib.pyplot as plt
plt.style.use('dark_background')
import glob
from skimage.color import rgb2lab
from skimage.transform import resize
from collections import namedtuple

```


```python
def current_name(folder, postfix):
    import time
    timestr = time.strftime("%y%m%d_%H%M%S")
    file_name = os.path.join('.',folder,timestr + postfix)
    print(file_name)
    return file_name
```


```python
import numpy as np
import pandas as pd
```


```python
np.random.seed(2019)
%matplotlib inline
```


```python
import os
os.listdir("./")
```




    ['.ipynb',
     '.ipynb_checkpoints',
     '190729_resnet50_transfer.ipynb',
     '190817_preprocessing-Copy1.ipynb',
     '190817_preprocessing.ipynb',
     '190817_VGG16.ipynb',
     '190819_inception_resnet2-v1.ipynb',
     '190821_resnet.ipynb',
     'Atos2019-ver1-Copy1.ipynb',
     'Atos2019-ver1.ipynb',
     'EDA_starter_resnet50-Copy1.ipynb',
     'EDA_starter_resnet50.ipynb',
     'gpu_configProto.ipynb',
     'input',
     'Intro APTOS Diabetic Retinopathy_EDA_Starter.ipynb',
     'Keras baseline.ipynb',
     'MODEL',
     'out.png',
     'resnet50_baseline-Copy1.ipynb',
     'resnet50_baseline.ipynb',
     'save',
     'simpleCNN',
     'submission',
     'submission.csv',
     'test.png',
     'Untitled.ipynb',
     'Untitled1.ipynb',
     'Untitled2.ipynb',
     'WORKING']




```python
os.listdir("./input/models")
```




    ['DenseNet-BC-121-32-no-top.h5',
     'DenseNet-BC-121-32.h5',
     'DenseNet-BC-161-48-no-top.h5',
     'DenseNet-BC-161-48.h5',
     'DenseNet-BC-169-32-no-top.h5',
     'DenseNet-BC-169-32.h5',
     'inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5',
     'inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5',
     'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
     'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5']




```python
from keras.models import Model
from keras import optimizers, applications
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
```


```python
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
```

    [name: "/device:CPU:0"
    device_type: "CPU"
    memory_limit: 268435456
    locality {
    }
    incarnation: 4712803198503752658
    , name: "/device:GPU:0"
    device_type: "GPU"
    memory_limit: 6682591232
    locality {
      bus_id: 1
      links {
      }
    }
    incarnation: 8721406740711979622
    physical_device_desc: "device: 0, name: GeForce GTX 1070 Ti, pci bus id: 0000:07:00.0, compute capability: 6.1"
    ]
    


```python
train_dir = "./input/train_images"
df_train = pd.read_csv("./input/train.csv")
df_test = pd.read_csv("./input/test.csv")
df_train['path'] = df_train['id_code'].map(lambda x: os.path.join(train_dir, '{}.png'.format(x)))
```

#### EDA (Explanatory Data Analysis)


```python
df = df_train
df.shape
```




    (3662, 3)




```python
df['diagnosis'].value_counts().plot(kind='bar')
plt.title('Samples Per Class')
```




    Text(0.5, 1.0, 'Samples Per Class')




![png](output_12_1.png)



```python
import seaborn as sns
f, ax = plt.subplots(figsize=(8, 6))
ax = sns.countplot(x='diagnosis', data=df_train, palette='viridis')
sns.despine()
plt.show()
```


![png](output_13_0.png)



```python
df['diagnosis'].value_counts().plot(kind='pie')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x213cf5e9e48>




![png](output_14_1.png)



```python
NUM_CLASSES = df_train['diagnosis'].nunique()
print(NUM_CLASSES)
```

    5
    


```python
df_train.head()
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
      <th>id_code</th>
      <th>diagnosis</th>
      <th>path</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>000c1434d8d7</td>
      <td>2</td>
      <td>./input/train_images\000c1434d8d7.png</td>
    </tr>
    <tr>
      <th>1</th>
      <td>001639a390f0</td>
      <td>4</td>
      <td>./input/train_images\001639a390f0.png</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0024cdab0c1e</td>
      <td>1</td>
      <td>./input/train_images\0024cdab0c1e.png</td>
    </tr>
    <tr>
      <th>3</th>
      <td>002c21358ce6</td>
      <td>0</td>
      <td>./input/train_images\002c21358ce6.png</td>
    </tr>
    <tr>
      <th>4</th>
      <td>005b95c28852</td>
      <td>0</td>
      <td>./input/train_images\005b95c28852.png</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3662 entries, 0 to 3661
    Data columns (total 3 columns):
    id_code      3662 non-null object
    diagnosis    3662 non-null int64
    path         3662 non-null object
    dtypes: int64(1), object(2)
    memory usage: 85.9+ KB
    


```python
df_train['diagnosis']=df_train['diagnosis'].astype('str')
```


```python
df_train['id_file'] = df_train['id_code'] + '.png'
```


```python
os.listdir(train_dir)[0:10]
```




    ['000c1434d8d7.png',
     '001639a390f0.png',
     '0024cdab0c1e.png',
     '002c21358ce6.png',
     '005b95c28852.png',
     '0083ee8054ee.png',
     '0097f532ac9f.png',
     '00a8624548a9.png',
     '00b74780d31d.png',
     '00cb6555d108.png']




```python
#sns.set_style('white')
plt.style.use('dark_background')
#plt.style.use('ggplot')
import cv2
count = 1
plt.figure(figsize=[20, 20])
for img_name in df_train['path'][0:15]:
    img = cv2.imread(img_name)
    plt.subplot(5,5,count)
    plt.imshow(img)
    plt.title('Image %s' % count)
    count += 1
    

```


![png](output_21_0.png)



```python
# from PIL import Image

# img = Image.open(df_train['path'][1])
# width, height = img.size
# print(width, height)
# img.show()
# plt.imshow(np.asarray(img))
```


```python
from PIL import Image
plt.figure(figsize=[20, 20])
count = 1
for img_name in df_train['path'][0:15]:
    img = Image.open(img_name)
    plt.subplot(5,5,count)
    plt.imshow(img)
    plt.title('Image %s' % count)
    count += 1
```


![png](output_23_0.png)


### Model Parameters


```python
# Model parameters
BATCH_SIZE = 64
EPOCHS = 40
WARMUP_EPOCHS = 1
LEARNING_RATE = 1e-4
WARMUP_LEARNING_RATE = 1e-3
HEIGHT = 256
WIDTH = 256
CHANNEL = 3
N_CLASSES = df_train['diagnosis'].nunique()

ES_PATIENCE = 5
RLROP_PATIENCE = 3
DECAY_DROP = 0.5



```


```python
# Preprocess data
df_test['id_file'] = df_test['id_code'].apply(lambda x: x + '.png')
df_test.head()
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
      <th>id_code</th>
      <th>id_file</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0005cfc8afb6</td>
      <td>0005cfc8afb6.png</td>
    </tr>
    <tr>
      <th>1</th>
      <td>003f0afdcd15</td>
      <td>003f0afdcd15.png</td>
    </tr>
    <tr>
      <th>2</th>
      <td>006efc72b638</td>
      <td>006efc72b638.png</td>
    </tr>
    <tr>
      <th>3</th>
      <td>00836aaacf06</td>
      <td>00836aaacf06.png</td>
    </tr>
    <tr>
      <th>4</th>
      <td>009245722fa4</td>
      <td>009245722fa4.png</td>
    </tr>
  </tbody>
</table>
</div>




```python
### Data Generator
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255.,
                                   validation_split=0.2,
                                   horizontal_flip=True)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=df_train,
    directory=train_dir,
    x_col='id_file',
    y_col='diagnosis',
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    target_size=(HEIGHT, WIDTH),
    subset='training')

valid_generator = train_datagen.flow_from_dataframe(
    dataframe=df_train,
    directory=train_dir,
    x_col='id_file',
    y_col='diagnosis',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    target_size=(HEIGHT, WIDTH),
    subset='validation')

test_datagen = ImageDataGenerator(rescale=1./255.)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=df_test,
    directory="./input/test_images/",
    x_col='id_file',
    target_size=(HEIGHT, WIDTH),
    batch_size=1,
    shuffle=False,
    class_mode=None)

```

    Found 2930 validated image filenames belonging to 5 classes.
    Found 732 validated image filenames belonging to 5 classes.
    Found 1928 validated image filenames.
    


```python
# from keras import optimizers, applications

# def create_model(input_shape, n_out):
# #    Resnet
# #    input_tensor = Input(shape=input_shape)
# #     base_model = applications.ResNet50(weights=None, 
# #                                        include_top=False,
# #                                        input_tensor=input_tensor)
# #     base_model.load_weights('./input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
#     base_model = applications.VGG16(weights='imagenet',
#                                    include_top=False,
#                                    input_shape=input_shape)
    
#     x = GlobalAveragePooling2D()(base_model.output)
#     x = Dropout(0.5)(x)
#     x = Dense(2048, activation='relu')(x)
#     x = Dropout(0.5)(x)
#     final_output = Dense(n_out, activation='softmax', name='final_output')(x)
#     model = Model(Input(input_shape), final_output)
    
#     return model
```


```python
# from keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input, Flatten
# from keras.models import Sequential
# from keras import optimizers, applications
# import tensorflow as tf

# vgg16 = applications.VGG16(weights='imagenet',
#                                    include_top=False,
#                                    input_shape=(HEIGHT, WIDTH, CHANNEL))

# vgg16.summary()
```


```python
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input, Flatten
from keras.models import Sequential
from keras import optimizers, applications
import tensorflow as tf

vgg16 = applications.VGG16(weights='imagenet',
                                   include_top=False,
                                   input_shape=(HEIGHT, WIDTH, CHANNEL))

vgg16.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_2 (InputLayer)         (None, 256, 256, 3)       0         
    _________________________________________________________________
    block1_conv1 (Conv2D)        (None, 256, 256, 64)      1792      
    _________________________________________________________________
    block1_conv2 (Conv2D)        (None, 256, 256, 64)      36928     
    _________________________________________________________________
    block1_pool (MaxPooling2D)   (None, 128, 128, 64)      0         
    _________________________________________________________________
    block2_conv1 (Conv2D)        (None, 128, 128, 128)     73856     
    _________________________________________________________________
    block2_conv2 (Conv2D)        (None, 128, 128, 128)     147584    
    _________________________________________________________________
    block2_pool (MaxPooling2D)   (None, 64, 64, 128)       0         
    _________________________________________________________________
    block3_conv1 (Conv2D)        (None, 64, 64, 256)       295168    
    _________________________________________________________________
    block3_conv2 (Conv2D)        (None, 64, 64, 256)       590080    
    _________________________________________________________________
    block3_conv3 (Conv2D)        (None, 64, 64, 256)       590080    
    _________________________________________________________________
    block3_pool (MaxPooling2D)   (None, 32, 32, 256)       0         
    _________________________________________________________________
    block4_conv1 (Conv2D)        (None, 32, 32, 512)       1180160   
    _________________________________________________________________
    block4_conv2 (Conv2D)        (None, 32, 32, 512)       2359808   
    _________________________________________________________________
    block4_conv3 (Conv2D)        (None, 32, 32, 512)       2359808   
    _________________________________________________________________
    block4_pool (MaxPooling2D)   (None, 16, 16, 512)       0         
    _________________________________________________________________
    block5_conv1 (Conv2D)        (None, 16, 16, 512)       2359808   
    _________________________________________________________________
    block5_conv2 (Conv2D)        (None, 16, 16, 512)       2359808   
    _________________________________________________________________
    block5_conv3 (Conv2D)        (None, 16, 16, 512)       2359808   
    _________________________________________________________________
    block5_pool (MaxPooling2D)   (None, 8, 8, 512)         0         
    =================================================================
    Total params: 14,714,688
    Trainable params: 14,714,688
    Non-trainable params: 0
    _________________________________________________________________
    


```python
for layer in vgg16.layers:
    layer.trainable = False


# generator new model 
model = Sequential()
model.add(vgg16)
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.5))
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(N_CLASSES, activation='softmax'))#, name='final_output'))

```


```python

```


```python


metric_list = [4]
optimizer = optimizers.Adam(lr=WARMUP_LEARNING_RATE)
model.compile(optimizer=optimizer, loss="categorical_crossentropy",  metrics=metric_list)
model.summary()

```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    vgg16 (Model)                (None, 8, 8, 512)         14714688  
    _________________________________________________________________
    global_average_pooling2d_1 ( (None, 512)               0         
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 512)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 2048)              1050624   
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 2048)              0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 5)                 10245     
    =================================================================
    Total params: 15,775,557
    Trainable params: 1,060,869
    Non-trainable params: 14,714,688
    _________________________________________________________________
    


```python
# STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
# STEP_SIZE_VALID = valid_generator.n//valid_generator.batch_size

# history_warmup = model.fit_generator(generator=train_generator,
#                                     steps_per_epoch=STEP_SIZE_TRAIN,
#                                     validation_data=valid_generator,
#                                     validation_steps=STEP_SIZE_VALID,
#                                     epochs=WARMUP_EPOCHS,
#                                     verbose=1).history
```


```python
# from keras.models import load_model

#model.save(current_name('WORKING','-resnet50.h5'))
# model = load_model("./WORKING/190818a_initialization.h5")
```


```python
# from numba import cuda
# cuda.select_device(0)
# cuda.close()
```

Fine-Tune the complete model


```python
STEP_SIZE_TRAIN = train_generator.n//BATCH_SIZE
STEP_SIZE_VALID = valid_generator.n//BATCH_SIZE

for layer in model.layers:
    layer.trainable = True

es = EarlyStopping(monitor='val_loss', mode='min', patience=ES_PATIENCE, 
                   restore_best_weights=True, verbose=1)
rlrop = ReduceLROnPlateau(monitor='val_loss',  mode='min', patience=RLROP_PATIENCE,
                          factor=DECAY_DROP, min_lr=1e-6, verbose=1)
#callback_list = [es, rlrop]
callback_list = [es]
optimizer = optimizers.Adam(lr=LEARNING_RATE)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=["accuracy"])
model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    vgg16 (Model)                (None, 8, 8, 512)         14714688  
    _________________________________________________________________
    global_average_pooling2d_1 ( (None, 512)               0         
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 512)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 2048)              1050624   
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 2048)              0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 5)                 10245     
    =================================================================
    Total params: 15,775,557
    Trainable params: 1,060,869
    Non-trainable params: 14,714,688
    _________________________________________________________________
    


```python
history_finetunning = model.fit_generator(generator=train_generator,
                                         steps_per_epoch=STEP_SIZE_TRAIN,
                                         validation_data=valid_generator,
                                         validation_steps=STEP_SIZE_VALID,
                                         epochs=EPOCHS,
                                         callbacks=callback_list,
                                         verbose=1).history
```

    Epoch 1/40
    45/45 [==============================] - 225s 5s/step - loss: 1.0485 - acc: 0.6345 - val_loss: 0.9389 - val_acc: 0.6392
    Epoch 2/40
    45/45 [==============================] - 220s 5s/step - loss: 0.9321 - acc: 0.6683 - val_loss: 0.9329 - val_acc: 0.6722
    Epoch 3/40
    45/45 [==============================] - 217s 5s/step - loss: 0.9352 - acc: 0.6703 - val_loss: 0.8669 - val_acc: 0.6811
    Epoch 4/40
    45/45 [==============================] - 218s 5s/step - loss: 0.8934 - acc: 0.6808 - val_loss: 0.8575 - val_acc: 0.6841
    Epoch 5/40
    45/45 [==============================] - 220s 5s/step - loss: 0.9043 - acc: 0.6739 - val_loss: 0.8751 - val_acc: 0.6841
    Epoch 6/40
    45/45 [==============================] - 220s 5s/step - loss: 0.8785 - acc: 0.6835 - val_loss: 0.8372 - val_acc: 0.6737
    Epoch 7/40
    45/45 [==============================] - 216s 5s/step - loss: 0.8908 - acc: 0.6729 - val_loss: 0.8583 - val_acc: 0.6901
    Epoch 8/40
    45/45 [==============================] - 217s 5s/step - loss: 0.8768 - acc: 0.6873 - val_loss: 0.8306 - val_acc: 0.6901
    Epoch 9/40
    45/45 [==============================] - 220s 5s/step - loss: 0.8670 - acc: 0.6925 - val_loss: 0.8955 - val_acc: 0.6841
    Epoch 10/40
    45/45 [==============================] - 220s 5s/step - loss: 0.8633 - acc: 0.6865 - val_loss: 0.8811 - val_acc: 0.6871
    Epoch 11/40
    45/45 [==============================] - 217s 5s/step - loss: 0.8733 - acc: 0.6882 - val_loss: 0.8270 - val_acc: 0.6976
    Epoch 12/40
    45/45 [==============================] - 221s 5s/step - loss: 0.8835 - acc: 0.6911 - val_loss: 0.9507 - val_acc: 0.5511
    Epoch 13/40
    45/45 [==============================] - 216s 5s/step - loss: 0.8303 - acc: 0.7033 - val_loss: 0.8736 - val_acc: 0.7141
    Epoch 14/40
    45/45 [==============================] - 219s 5s/step - loss: 0.8457 - acc: 0.6974 - val_loss: 0.9102 - val_acc: 0.6976
    Epoch 15/40
    45/45 [==============================] - 220s 5s/step - loss: 0.8603 - acc: 0.6875 - val_loss: 0.8138 - val_acc: 0.7021
    Epoch 16/40
    45/45 [==============================] - 218s 5s/step - loss: 0.8453 - acc: 0.6976 - val_loss: 0.8431 - val_acc: 0.6931
    Epoch 17/40
    45/45 [==============================] - 220s 5s/step - loss: 0.8563 - acc: 0.6895 - val_loss: 0.8805 - val_acc: 0.6572
    Epoch 18/40
    45/45 [==============================] - 217s 5s/step - loss: 0.8361 - acc: 0.7008 - val_loss: 0.8884 - val_acc: 0.6632
    Epoch 19/40
    45/45 [==============================] - 220s 5s/step - loss: 0.8468 - acc: 0.6869 - val_loss: 0.8468 - val_acc: 0.7066
    Epoch 20/40
    45/45 [==============================] - 219s 5s/step - loss: 0.8453 - acc: 0.6942 - val_loss: 0.8836 - val_acc: 0.7066
    Restoring model weights from the end of the best epoch
    Epoch 00020: early stopping
    


```python
# from keras.models import load_model

model.save(current_name('WORKING','-vgg16.h5'))
# model = load_model("./WORKING/190818a_initialization.h5")
```

    .\WORKING\190821_194602-vgg16.h5
    

### Model loss graph


```python
# history = {'loss': history_warmup['loss'] + history_finetunning['loss'],
#            'val_loss':history_warmup['val_loss'] + history_finetunning['val_loss'],
#            'acc': history_warmup['acc'] + history_finetunning['acc'],
#            'val_acc': history_warmup['val_acc'] + history_finetunning['val_acc']}
history = {'loss': history_finetunning['loss'],
           'val_loss':history_finetunning['val_loss'],
           'acc':history_finetunning['acc'],
          'val_acc': history_finetunning['val_acc']}



#sns.set_style('whitegrid')

fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', figsize=(12,8))

ax1.plot(history['loss'], label='Train loss')
ax1.plot(history['val_loss'], label='Validation loss')
ax1.legend(loc='best')
ax1.set_title('Loss')

ax2.plot(history['acc'], label='Train Accuracy')
ax2.plot(history['val_acc'], label='Validation accuracy')
ax2.legend(loc='best')
ax2.set_title('Accuracy')

plt.xlabel('Epochs')
sns.despine()
plt.show()





```


![png](output_42_0.png)



```python
# from numba import cuda
# cuda.select_device(0)
# cuda.close()

# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
# set_session(tf.Session(config=config))
```

### Model Evaluation


```python
complete_datagen = ImageDataGenerator(rescale=1./255)
complete_generator = complete_datagen.flow_from_dataframe(
        dataframe=df_train,
        directory="./input/train_images/",
        x_col='id_file',
        target_size=(HEIGHT, WIDTH),
        batch_size=1,
        shuffle=False,
        class_mode=None)
STEP_SIZE_COMPLETE = complete_generator.n//1
train_preds = model.predict_generator(complete_generator, steps=STEP_SIZE_COMPLETE)
train_preds = [np.argmax(pred) for pred in train_preds]

```

    Found 3662 validated image filenames.
    


```python
train_preds[0:10]
```




    [2, 2, 1, 0, 0, 2, 2, 2, 2, 1]




```python
from sklearn.metrics import confusion_matrix, cohen_kappa_score
labels = ['0 - No DR', '1 - Mild', '2 - Moderate', '3 - Severe', '4 - Proliferative DR']
cnf_matrix = confusion_matrix(df_train['diagnosis'].astype('int'), train_preds)
#cnf_matrix_norm = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
#df_cm = pd.DataFrame(cnf_matrix_norm, index=labels, columns=labels)
df_cm = pd.DataFrame(cnf_matrix, index=labels, columns=labels)
plt.figure(figsize=(16, 7))
sns.heatmap(df_cm, annot=True, fmt='.2f', cmap="Blues")
plt.show()
```


![png](output_47_0.png)



```python
print("Train Cohen Kappa score: %.3f" % cohen_kappa_score(train_preds, df_train['diagnosis'].astype('int'), weights='quadratic'))
```

    Train Cohen Kappa score: 0.696
    

### Apply model to test set and output prediction


```python
test_generator.reset()
STEP_SIZE_TEST = test_generator.n//test_generator.batch_size
preds = model.predict_generator(test_generator, steps=STEP_SIZE_TEST)
predictions = [np.argmax(pred) for pred in preds]
```


```python
filenames = test_generator.filenames
results = pd.DataFrame({'id_code':filenames, 'diagnosis':predictions})
results['id_code'] = results['id_code'].map(lambda x: str(x)[:-4])
results.to_csv('submission.csv',index=False)
results.head(10)
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
      <th>id_code</th>
      <th>diagnosis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0005cfc8afb6</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>003f0afdcd15</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>006efc72b638</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>00836aaacf06</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>009245722fa4</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>009c019a7309</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>010d915e229a</td>
      <td>2</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0111b949947e</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>01499815e469</td>
      <td>2</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0167076e7089</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
results.to_csv(current_name('submission', '-submission.csv'),index=False)
```
