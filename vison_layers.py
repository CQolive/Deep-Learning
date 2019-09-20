#from keras.models import load_model
#model = load_model('cats_and_dogs_small_2.h5')
#img_path = 'E:/代码/DL_Python/dogs-vs-cats/cats_and_dogs_small/test/cats/cat.1700.jpg'
#from keras.preprocessing import image#将图像预处理为一个4D张量
#import numpy as np
#
#img = image.load_img(img_path,target_size=(150,150))
#img_tensor = image.img_to_array(img)
#img_tensor = np.expand_dims(img_tensor,axis=0)
#img_tensor /= 255.  #训练模型的输入数据都用这种方法预处理
#
##其形状为（1，150，150，3）
##print(img_tensor.shape)
##显示测试图像
#import matplotlib.pyplot as plt
#
##plt.imshow(img_tensor[0])
##plt.show()
#
##用一个输入张量和一个输出张量列表将模型实例化
#from keras import models
##提取前八层输出
#layer_outputs = [layer.output for layer in model.layers[:8]]
#activation_model = models.Model(inputs=model.input,outputs=layer_outputs)
##以预测模式运行模型
#activations = activation_model.predict(img_tensor)
#first_layer_activation = activations[0]
#import matplotlib.pyplot as plt
##将第 4 个通道可视化
##plt.matshow(first_layer_activation[0, :, :,4], cmap='viridis')
##将第 7 个通道可视化
##plt.matshow(first_layer_activation[0, :, :,7], cmap='viridis')
##plt.show()
#
##将每个中间层激活的所有通道可视化
#layer_names = []
#for layer in model.layers[:8]:
#    layer_names.append(layer.name)
#
#images_per_row = 16
#
#for layer_name,layer_activation in zip(layer_names,activations):
#    n_features = layer_activation.shape[-1]  #特征图中的特征数量
#    
#    size = layer_activation.shape[1] #特征图的形状为(1,size,size,n_features)
#
#    n_cols = n_features // images_per_row
#    display_grid = np.zeros((size * n_cols,images_per_row * size))
#    for col in range(n_cols):
#        for row in range(images_per_row):
#            channel_image = layer_activation[0,
#            :,:,
#            col * images_per_row + row]
#            channel_image -= channel_image.mean()
#            channel_image /= channel_image.std()
#            channel_image *= 64
#            channel_image += 128
#            channel_image = np.clip(channel_image,0,255).astype('uint8')
#            display_grid[col * size : (col + 1) * size,
#                        row * size : (row + 1) * size] = channel_image
#    scale = 1./size
#    plt.figure(figsize=(scale * display_grid.shape[1],
#                        scale * display_grid.shape[0]))
#    plt.title(layer_name)
#    plt.grid(False)
#    plt.imshow(display_grid,aspect='auto',cmap='viridis')
#plt.show()
            


#可视化卷积神经网络的过滤器,从空白输入图像开始，将梯度下降应用
#于卷积神经网络输入图像的值，其目的是让某个过滤器的响应最大化

## 为过滤器的可视化定义损失张量
#from keras.applications import VGG16
from keras import backend as K
#model = VGG16(weights='imagenet',include_top=False)
#layer_name = 'block3_conv1'
#filter_index = 0
#layer_output = model.get_layer(layer_name).output
#loss = K.mean(layer_output[:,:,:,filter_index])
#
##获取损失相对于输入的梯度
#grads = K.gradients(loss,model.input)[0]
##梯度标准化技巧
#grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
##给定 Numpy 输入值，得到 Numpy 输出值
#iterate = K.function([model.input],[loss,grads])
#
#import numpy as np
#loss_value,grads_value = iterate([np.zeros((1,150,150,3))])
#
##　通过随机梯度下降让损失最大化
#input_image_data = np.random.random((1,150,150,3)) * 20 + 128. #从一张带有噪声的灰度图像开始
#step = 1.#每次梯度更新的步长
#for i in range(40):
#    loss_value,grads_value = iterate([input_image_data])
#    input_image_data += grads_value * step
#
##将张量转换为有效图像的实用函数
#def deprocess_image(x):
#    x -= x.mean()
#    x /= (x.std() + 1e-5)
#    x *= 0.1
#
#    x += 0.5
#    x = np.clip(x,0,1)
#
#    x *= 255
#    x = np.clip(x,0,255).astype('uint8')
#    return x
#
##生成过滤器可视化的函数
#def generate_pattern(layer_name,filter_index,size=150):
#    layer_output = model.get_layer(layer_name).output
#    loss = K.mean(layer_output[:,:,:,filter_index])
#
#    grads = K.gradients(loss,model.input)[0]
#    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
#
#    iterate = K.function([model.input],[loss,grads])
#
#    input_image_data = np.random.random((1,size,size,3)) * 20 +128
#
#    step = 1.
#    for i in range(40):
#        loss_value,grads_value = iterate([input_image_data])
#        input_image_data += grads_value * step
#    
#    img = input_image_data[0]
#    return deprocess_image(img)
#
import matplotlib.pyplot as plt
##block3_conv1 层第 0 个过滤器响应的是波尔卡点（ polka-dot）图案
##plt.imshow(generate_pattern('block3_conv1', 0))
##plt.show()
#for i,layer_name in enumerate(['block1_conv1','block2_conv1','block3_conv1','block4_conv1']):
#    size = 64
#    margin = 5
#    #空图像
#    results = np.zeros((8 * size + 7 * margin,8 * size + 7 * margin,3))
#    
#    for i in range(8):
#        for j in range(8):
#            #生成layer_name层第i+(j*8)个过滤器的模式
#            filter_img = generate_pattern(layer_name,i + (j * 8),size=size)
#            #将结果放到results网格第(i,j)个方块中
#            horizontal_start = i * size + i * margin
#            horizontal_end = horizontal_start + size
#            vertical_start = j * size + j * margin
#            vertical_end = vertical_start + size
#            results[horizontal_start:horizontal_end,
#                    vertical_start:vertical_end,:] = filter_img
#    plt.figure(figsize=(20,20))
#    results = results/255.
#    plt.title(layer_name)
#    plt.imshow(results)
#plt.show()
#

from keras.applications.vgg19 import VGG19

model = VGG19(weights='imagenet')

from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input,decode_predictions
import numpy as np

img_path ='E:/代码/DL_Python/789.png'
#大小为224X244的Python图像库
img = image.load_img(img_path,target_size=(224,224))
#形状为(224,224,3)的float32格式的Numpy数组
x = image.img_to_array(img)
#添加一个维度。将数组转换为(1,224,224,3)形状的批量
x = np.expand_dims(x,axis=0)
#对批量进行预处理(按通道进行颜色标准化)
x = preprocess_input(x)

preds = model.predict(x)
print('Predicted:',decode_predictions(preds,top=3)[0])
print(np.argmax(preds[0]))

#应用Grad-CAM算法
#预测向量中的"非洲象"元素
african_elephant_output = model.output[:,386]
#模型最后一个卷积层
last_conv_layer = model.get_layer('block5_conv4')
#"非洲象"类别相对于block5_conv3输出特征图的梯度
grads = K.gradients(african_elephant_output,last_conv_layer.output)[0]
#形状为(512,)的向量，每个元素是特定特征图通道的梯度平均大小
pooled_grads = K.mean(grads,axis=(0,1,2))
#访问刚刚定义的量：对于给定的样本图像，pooled_grads和block5_conv3层的输出特征图
iterate = K.function([model.input],[pooled_grads,last_conv_layer.output[0]])
#对于两个大象的样本图像，这两个量都是Numpy数组
pooled_grads_value,conv_layer_output_value = iterate([x])
for i in range(512):
#将特征图数组的每个通道乘以“这个通道对‘大象’类别的重要程度”
    conv_layer_output_value[:,:,i] *= pooled_grads_value[i]
#得到的特征图的逐通道平均值即为类激活的热力图
heatmap = np.mean(conv_layer_output_value,axis=-1)
#热力图后处理
heatmap = np.maximum(heatmap,0)
heatmap /= np.max(heatmap)
#plt.matshow(heatmap)
#plt.show()

import cv2
#用cv2加载原始图像
img = cv2.imread('789.png')
#img = image.load_img(img_path,target_size=(img.shape[1],img.shape[0]))
#将热力图的大小调整为与原始图像相同
heatmap = cv2.resize(heatmap,(img.shape[1],img.shape[0]))
#将热力图转化为RGB
heatmap = np.uint8(255 * heatmap)
#将热力图应用于原始图
heatmap = cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)
#这里的0.4是热力图强度因子
superimposed_img = heatmap * 0.5 + img
print(superimposed_img)
plt.imshow(superimposed_img/255.)
plt.show()

