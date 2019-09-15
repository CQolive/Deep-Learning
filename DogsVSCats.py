import os,shutil
original_dataset_dir = 'E:/代码/DL_Python/dogs-vs-cats/train/train'
#
base_dir = 'E:/代码/DL_Python/dogs-vs-cats/cats_and_dogs_small'
#os.mkdir(base_dir)
##划分训练，验证，测试目录
train_dir = os.path.join(base_dir,'train')
#os.mkdir(train_dir)
validation_dir = os.path.join(base_dir,'validation')
#os.mkdir(validation_dir)
test_dir = os.path.join(base_dir,'test')
#os.mkdir(test_dir)
##猫和狗的训练集
train_cats_dir = os.path.join(train_dir,'cats')
#os.mkdir(train_cats_dir)
train_dogs_dir = os.path.join(train_dir,'dogs')
#os.mkdir(train_dogs_dir)
##猫和狗的验证集
validation_cats_dir = os.path.join(validation_dir,'cats')
#os.mkdir(validation_cats_dir)
validation_dogs_dir = os.path.join(validation_dir,'dogs')
#os.mkdir(validation_dogs_dir)
##猫和狗的测试集
test_cats_dir = os.path.join(test_dir,'cats')
#os.mkdir(test_cats_dir)
test_dogs_dir = os.path.join(test_dir,'dogs')
#os.mkdir(test_dogs_dir)
##将前 1000 张猫的图像复制到 train_cats_dir
#fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
#for fname in fnames:
#    src = os.path.join(original_dataset_dir,fname)
#    dst = os.path.join(train_cats_dir,fname)
#    shutil.copyfile(src,dst)
#
##将下面500张猫复制到validation_cats_dir
#fnames = ['cat.{}.jpg'.format(i) for i in range(1000,1500)]
#for fname in fnames:
#    src = os.path.join(original_dataset_dir,fname)
#    dst = os.path.join(validation_cats_dir,fname)
#    shutil.copyfile(src,dst)
#
##将下面500张猫复制到test_cats_dir
#fnames = ['cat.{}.jpg'.format(i) for i in range(1500,2000)]
#for fname in fnames:
#    src = os.path.join(original_dataset_dir,fname)
#    dst = os.path.join(test_cats_dir,fname)
#    shutil.copyfile(src,dst)
##将前 1000 张狗的图像复制到 train_dogs_dir
#fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
#for fname in fnames:
#    src = os.path.join(original_dataset_dir,fname)
#    dst = os.path.join(train_dogs_dir,fname)
#    shutil.copyfile(src,dst)
##将下面500张狗复制到validation_dogs_dir
#fnames = ['dog.{}.jpg'.format(i) for i in range(1000,1500)]
#for fname in fnames:
#    src = os.path.join(original_dataset_dir,fname)
#    dst = os.path.join(validation_dogs_dir,fname)
#    shutil.copyfile(src,dst)
##将下面500张狗复制到test_dogs_dir
#fnames = ['cat.{}.jpg'.format(i) for i in range(1500,2000)]
#for fname in fnames:
#    src = os.path.join(original_dataset_dir,fname)
#    dst = os.path.join(test_dogs_dir,fname)
#    shutil.copyfile(src,dst)

#print("total training cat images：",len(os.listdir(train_cats_dir)))
#print("total training cat images：",len(os.listdir(train_dogs_dir)))
#print("total training cat images：",len(os.listdir(validation_cats_dir)))
#print("total training cat images：",len(os.listdir(validation_dogs_dir)))
#print("total training cat images：",len(os.listdir(test_cats_dir)))
#print("total training cat images：",len(os.listdir(test_dogs_dir)))

##猫狗分类小型卷积神经网络
from keras import layers
from keras import models
#
#model = models.Sequential()
#model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
#model.add(layers.MaxPool2D((2,2)))
#model.add(layers.Conv2D(64,(3,3),activation='relu'))
#model.add(layers.MaxPool2D((2,2)))
#model.add(layers.Conv2D(128,(3,3),activation='relu'))
#model.add(layers.MaxPool2D((2,2)))
#model.add(layers.Conv2D(128,(3,3),activation='relu'))
#model.add(layers.MaxPool2D((2,2)))
#model.add(layers.Flatten())
#model.add(layers.Dense(512,activation='relu'))
#model.add(layers.Dense(1,activation='sigmoid'))
#print(model.summary())
#
from keras import optimizers
#model.compile(loss='binary_crossentropy',
#            optimizer=optimizers.RMSprop(lr=1e-4),
#            metrics=['acc']
#)
#
##使用 ImageDataGenerator 从目录中读取图像
from keras.preprocessing.image import ImageDataGenerator
##将所有图片乘以1/255缩放
#train_datagen = ImageDataGenerator(rescale=1./255)
#test_datagen = ImageDataGenerator(rescale=1./255)
#
#train_generator = train_datagen.flow_from_directory(
#    train_dir,
#    target_size=(150,150),#将所有图片调整到150*150
#    batch_size=20,
#    class_mode='binary')
#validation_generator = test_datagen.flow_from_directory(
#    validation_dir,
#    target_size=(150,150),
#    batch_size=20,
#    class_mode='binary'
#)
#history = model.fit_generator(
#    train_generator,
#    steps_per_epoch=100,
#    epochs=30,
#    validation_data=validation_generator,
#    validation_steps=50
#)
##保存模型
#model.save('cats_and_dogs_small_1.h5')
#
##绘制训练过程中的损失曲线和精度曲线
import matplotlib.pyplot as plt
#
#acc = history.history['acc']
#val_acc = history.history['val_acc']
#loss = history.history['loss']
#val_loss = history.history['val_loss']
#
#epochs = range(1,len(acc) + 1)
#
#plt.plot(epochs,acc,'bo',label='Training acc')
#plt.plot(epochs,val_acc,'b',label='Validation acc')
#plt.title('Training and validation accuracy')
#
#plt.figure()
#
#plt.plot(epochs,loss,'bo',label='Training loss')
#plt.plot(epochs,val_loss,'b',label='Validation loss')
#plt.title('Training and validation loss')
#
#plt.show()

##过拟合
##绘制训练过程中的损失曲线和精度曲线
#datagen = ImageDataGenerator(
#    rotation_range=40,
#    width_shift_range=0.2,
#    height_shift_range=0.2,
#    shear_range=0.2,
#    horizontal_flip=True,
#    fill_mode='nearest'
#)

##显示几个随机增强后的训练图像
#from keras.preprocessing import image
#fnames = [os.path.join(train_cats_dir,fname) for fname in os.listdir(train_cats_dir)]
#img_path = fnames[3]
#img = image.load_img(img_path,target_size=(150,150))
#x = image.img_to_array(img)
#x = x.reshape((1,) + x.shape)
#
#i = 0
#for batch in datagen.flow(x,batch_size=1):
#    plt.figure(i)
#    imgplot = plt.imshow(image.array_to_img(batch[0]))
#    i += 1
#    if i % 4 == 0:
#        break
#plt.show()
#

##定义一个包含 dropout 的新卷积神经网络
#model = models.Sequential()
#model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
#model.add(layers.MaxPool2D((2,2)))
#model.add(layers.Conv2D(64,(3,3),activation='relu'))
#model.add(layers.MaxPool2D((2,2)))
#model.add(layers.Conv2D(128,(3,3),activation='relu'))
#model.add(layers.MaxPool2D((2,2)))
#model.add(layers.Conv2D(128,(3,3),activation='relu'))
#model.add(layers.MaxPool2D((2,2)))
#model.add(layers.Flatten())
#model.add(layers.Dropout(0.5))
#model.add(layers.Dense(512,activation='relu'))
#model.add(layers.Dense(1,activation='sigmoid'))
#
#model.compile(loss='binary_crossentropy',
#        optimizer=optimizers.RMSprop(lr=1e-4),
#        metrics=['acc']
#)
#
##利用数据增强生成器训练卷积神经网络
#train_datagen = ImageDataGenerator(
#    rescale=1./255,
#    rotation_range=40,
#    width_shift_range=0.2,
#    height_shift_range=0.2,
#    shear_range=0.2,
#    zoom_range=0.2,
#    horizontal_flip=True
#)
##验证集不能增强
#test_datagen = ImageDataGenerator(rescale=1./255)
#
#train_generator = train_datagen.flow_from_directory(
#    train_dir,
#    target_size=(150,150),
#    batch_size=32,
#    class_mode='binary'
#)
#
#validation_generator = test_datagen.flow_from_directory(
#    validation_dir,
#    target_size=(150,150),
#    batch_size=32,
#    class_mode='binary'
#)
#
#history = model.fit_generator(
#    train_generator,
#    steps_per_epoch=100,
#    epochs=100,
#    validation_data=validation_generator,
#    validation_steps=50
#)
#
#model.save('cats_and_dogs_small_2.h5')
#acc = history.history['acc']
#val_acc = history.history['val_acc']
#loss = history.history['loss']
#val_loss = history.history['val_loss']
#
#epochs = range(1,len(acc) + 1)
#
#plt.plot(epochs,acc,'bo',label='Training acc')
#plt.plot(epochs,val_acc,'b',label='Validation acc')
#plt.title('Training and validation accuracy')
#
#plt.figure()
#
#plt.plot(epochs,loss,'bo',label='Training loss')
#plt.plot(epochs,val_loss,'b',label='Validation loss')
#plt.title('Training and validation loss')
#
#plt.show()



#将 VGG16 卷积基实例化
from keras.applications import VGG16
conv_base = VGG16(weights='imagenet',
        include_top=False,
        input_shape=(150,150,3)
)
##使用预训练的卷积基提取特征
#import numpy as np
#datagen = ImageDataGenerator(rescale=1./255)
#batch_size = 20
#def extract_features(directory,sample_count):
#    features = np.zeros(shape=(sample_count,4,4,512))
#    labels = np.zeros(shape=(sample_count))
#    generator = datagen.flow_from_directory(
#        directory,
#        target_size=(150,150),
#        batch_size=batch_size,
#        class_mode='binary'
#    )
#    i = 0
#    for inputs_batch,labels_batch in generator:
#        features_batch = conv_base.predict(inputs_batch)
#        features[i * batch_size : (i + 1) * batch_size] = features_batch
#        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
#        i += 1
#        if i * batch_size >= sample_count:
#            break
#    return features,labels
#train_features,train_labels = extract_features(train_dir,2000)
#validation_features,validation_labels = extract_features(validation_dir,1000)
#test_features,test_labels = extract_features(test_dir,1000)
##目前，提取的特征形状为 (samples, 4, 4, 512)。我们要将其输入到密集连接分类器中，
##所以首先必须将其形状展平为 (samples, 8192)
#train_features = np.reshape(train_features,(2000,4 * 4 * 512))
#validation_features = np.reshape(validation_features,(1000,4 * 4 * 512))
#test_features = np.reshape(test_features,(1000,4 * 4 * 512))
#
##定义并训练密集连接分类器
#model = models.Sequential()
#model.add(layers.Dense(256,activation='relu',input_dim = 4 * 4 *512))
#model.add(layers.Dropout(0.5))
#model.add(layers.Dense(1,activation = 'sigmoid'))
#
#model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
#    loss='binary_crossentropy',
#    metrics=['acc']
#)
#
#history = model.fit(train_features,train_labels,
#            epochs=30,
#            batch_size=20,
#            validation_data=(validation_features,validation_labels)
#)
##绘制结果
def plofigure(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1,len(acc)+1)
    plt.plot(epochs,acc,'bo',label='Training acc')
    plt.plot(epochs,val_acc,'b',label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    
    plt.figure()
    
    plt.plot(epochs,loss,'bo',label='Training loss')
    plt.plot(epochs,val_loss,'b',label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()
    

#在卷积基上添加一个密集连接分类器
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

#利用冻结的卷积基端到端地训练模型
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(rescale=1./255) #不能增强验证数据
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary'
)
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary'
)
model.compile(loss='binary_crossentropy',
            optimizer=optimizers.RMSprop(lr=2e-5),
            metrics=(['acc'])
)
#history = model.fit_generator(
#    train_generator,
#    steps_per_epoch=100,
#    epochs=30,
#    validation_data=validation_generator,
#    validation_steps=50
#)
#plofigure(history)

#冻结直到某一层
conv_base.trainable = True
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
#微调模型
model.compile(loss='binary_crossentropy',
            optimizer=optimizers.RMSprop(lr=1e-5),
            metrics=['acc']
    )
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=50
)
#使曲线变得平滑
def smooth_curve(points,factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points
epochs = range(1,len(acc)+1)
plt.plot(epochs,smooth_curve(acc), 'bo', label='Smoothed training acc')
plt.plot(epochs,smooth_curve(val_acc),'b',label='Smoothed validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs,
smooth_curve(loss), 'bo', label='Smoothed training loss')
plt.plot(epochs,
smooth_curve(val_loss), 'b', label='Smoothed validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

#最终评估模型
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary'
)
test_loss,test_acc = model.evaluate_generator(test_generator,steps=50)
print('test acc: ',test_acc)