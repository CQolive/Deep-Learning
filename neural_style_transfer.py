#定义初始变量
from keras.preprocessing.image import load_img,img_to_array
#想要变换的图像路径
target_image_path = '2.png'
#风格图像路径
style_reference_image_path = '1.png'
#生成图像的尺寸
width,height = load_img(target_image_path).size
img_height = 400
img_width = int(width * img_height / height)
#辅助函数
import numpy as np
from keras.applications import vgg19
def preprocess_image(image_path):
    img = load_img(image_path,target_size=(img_height,img_width))
    img = img_to_array(img)
    img = np.expand_dims(img,axis=0)
    img = vgg19.preprocess_input(img)
    return img
def deprocess_image(x):
    #vgg19.preprocess_input的作用是减去ImageNet的平均像素值，
    # 使其中心为0.这里相当于vgg19.preprocess_input的逆操作
    x[:,:,0] += 103.939
    x[:,:,1] += 116.779
    x[:,:,2] += 123.68
    #将图像由BGR格式转换为RGB格式，这也是vgg19.preprocess_input逆操作的一部分
    x = x[:,:,::-1]
    x = np.clip(x,0,255).astype('uint8')
    return x
#加载预训练的 VGG19 网络，并将其应用于三张图像
from keras import backend as K
target_image = K.constant(preprocess_image(target_image_path))
style_reference_image = K.constant(preprocess_image(style_reference_image_path))
#这个占位符用于保存生成图像
combination_image = K.placeholder((1,img_height,img_width,3))
#将三张图像合并成为一个批量
input_tensor = K.concatenate([target_image,style_reference_image,combination_image],axis=0)
#利用三张图像组成的批量作为输入来构建VGG19网络。加载模型将使用预训练的ImageNet权重
model = vgg19.VGG19(input_tensor=input_tensor,weights='imagenet',include_top=False)
print('Model loaded.')

#内容损失
def content_loss(base,combination):
    return K.sum(K.square(combination - base))
#风格损失
def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x,(2,0,1)))
    gram = K.dot(features,K.transpose(features))
    return gram
def style_loss(style,combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_height * img_width
    return K.sum(K.square(S -C)) / (4. * (channels ** 2) * (size ** 2))
#总变差损失
def total_variation_loss(x):
    a = K.square(
        x[:,:img_height - 1, :img_width - 1,:] -
        x[:,1:,:img_width - 1, :]
    )
    b = K.square(
        x[:,:img_height - 1,:img_width - 1,:] -
        x[:,:img_height - 1,1:,:]
    )
    return K.sum(K.pow(a + b,1.25))
#定义需要最小化的最终损失
#将层的名称映射为激活张量的字典
outputs_dict = dict([(layer.name,layer.output) for layer in model.layers])
#用于内容损失的层
content_layer = 'block5_conv2'
#用于风格损失的层
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']
#损失分量的加权平均所使用的权重
total_variation_weight = 1e-4
style_weight = 1.
content_weight = 0.025
#在定义损失时将所有分量添加到这个标量变量中
#添加内容损失
loss = K.variable(0.)
layer_features = outputs_dict[content_layer]
target_image_features = layer_features[0,:,:,:]
combination_features = layer_features[2,:,:,:]
loss += content_weight * content_loss(target_image_features,combination_features)
#添加每个目标层的风格损失分量
for layer_name in style_layers:
    layer_features = outputs_dict[layer_name]
    style_reference_features = layer_features[1,:,:,:]
    combination_features = layer_features[2,:,:,:]
    sl = style_loss(style_reference_features,combination_features)
    loss += (style_weight / len(style_layers)) * sl
#添加总变差损失
loss += total_variation_weight * total_variation_loss(combination_image)

#设置梯度下降过程
#获取损失相对于生成图像的梯度
grads = K.gradients(loss,combination_image)[0]
#用于获取当前损失值和当前梯度值的函数
fetch_loss_and_grads = K.function([combination_image],[loss,grads])
#这个类将fetch_loss_and_grads包装起来，让你可以利用两个单独的方法调用来获取损失和梯度，
#这是我们要使用SciPy优化器所要求的
class Evaluator(object):
    def __init__(self):
        self.loss_value = None
        self.grads_values = None
    def loss(self,x):
        assert self.loss_value is None
        x = x.reshape((1,img_height,img_width,3))
        outs = fetch_loss_and_grads([x])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype('float64')
        self.loss_value = loss_value
        self.grads_values = grad_values
        return self.loss_value
    def grads(self,x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grads_values)
        self.loss_value = None
        self.grads_values = None
        return grad_values
evaluator = Evaluator()
#风格迁移循环
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave
import time

result_prefix = 'my_result'
iterations = 20

#初始状态：目标图像
x = preprocess_image(target_image_path)
#将图像展平，因为scipy.optimize.fmin_l_bfgs_b只能处理展平的向量
x = x.flatten()
for i in range(iterations):
    print('Start of iteration',i)
    start_time = time.time()
    #对生成图像的像素运行L-BFGS最优化，以将神经风格损失最小化。
    #注意，必须将计算损失的函数何人计算梯度的函数作为两个单独的参数传入
    x,min_val,info = fmin_l_bfgs_b(
        evaluator.loss,
        x,
        fprime=evaluator.grads,
        maxfun=20
    )
    print('Current loss value:',min_val)
    #保存当前的生成图像
    img = x.copy().reshape((img_height,img_width,3))
    img = deprocess_image(img)
    fname = result_prefix + '_at_iteration_%d.png' % i
    imsave(fname,img)
    print('Image saved as',fname)
    end_time = time.time()
    print('Iteration %d completed in %ds' % (i,end_time - start_time))

