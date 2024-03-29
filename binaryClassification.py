#加载 IMDB 数据集
from keras.datasets import imdb
#参数 num_words=10000 的意思是仅保留训练数据中前 10 000 个最常出现的单词
(train_data,train_labels),(test_data,test_labels) = imdb.load_data(num_words=1000)
#某条评论迅速解码为英文单词
word_index = imdb.get_word_index()
reverse_word_index = dict(
    [(value,key) for (key,value) in word_index.items()])
decoded_review = ' '.join(
    [reverse_word_index.get(i-3,'?') for i in train_data[0]]
)

#将整数序列编码为二进制矩阵
import numpy as np
def vectorize_sequences(sequences,dimension=10000):
    results = np.zeros((len(sequences),dimension))
    for i,sequence in enumerate(sequences):
        results[i,sequence] = 1.
    return results
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')


#模型定义
from keras import models
from keras import layers
model = models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))
#编译模型
model.compile(optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['accuracy'])
##配置优化器
from keras import optimizers
model.compile(optimizer=optimizers.RMSprop(lr=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
)
#使用自定义的损失和指标
from keras import losses
from keras import metrics
model.compile(optimizer=optimizers.RMSprop(lr=0.001),
                loss=losses.binary_crossentropy,
                metrics=[metrics.binary_accuracy]
                )
#留出验证集
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

#训练模型
'''
 512 个样本组成的小批量，将模型训练 20 个轮次（即对 x_train 和 y_train 两
个张量中的所有样本进行 20 次迭代）。与此同时，你还要监控在留出的 10 000 个样本上的损失
和精度。通过将验证数据传入 validation_data 参数来完成。
'''
model.compile(optimizer='rmsprop',
                loss='binary_crossentropy',
                metrics=['acc']
)
history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=20,
    batch_size=512,
    validation_data=(x_val,y_val)
)
history_dict = history.history
print(history_dict.keys())

#绘制训练损失和验证损失
import matplotlib.pyplot as plt

history_dict =history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1,len(loss_values) + 1)

plt.figure(1)
plt.plot(epochs,loss_values,'bo',label='Training loss')
plt.plot(epochs,val_loss_values,'b',label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show

plt.figure(2)
acc = history_dict['acc']
val_acc = history_dict['val_acc']

plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'b',label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
#在第四轮后过拟合

#从头开始重新训练一个模型，训练4轮
model = models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))
model.compile(
    optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
model.fit(x_train,y_train,epochs=4,batch_size=512)
results = model.evaluate(x_test,y_test)

