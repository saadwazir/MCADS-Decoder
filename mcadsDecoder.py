
def read_config(file_name):
    config = {}
    with open(file_name, 'r') as file:
        for line in file:
            if not line.strip() or line.startswith('#'):
                continue
            key, value = line.strip().split('=')
            config[key] = value
    return config


config = read_config('config.txt')


train_images_patch_dir = config.get('train_images_patch_dir')
train_masks_patch_dir = config.get('train_masks_patch_dir')

test_images_patch_dir = config.get('test_images_patch_dir')
test_masks_patch_dir = config.get('test_masks_patch_dir')

image_full_test_directory = config.get('image_full_test_directory')
mask_full_test_directory = config.get('mask_full_test_directory')

gpu_device = config.get('gpu_device')

num_epochs = int(config.get('num_epochs',0))

batch_size = int(config.get('batch_size', 0))

imgz_size = int(config.get('imgz_size', 0))


training = config.get('training')
if (training == "True"):
    training = True
else:
    training = False



patch_img_size = int(config.get('patch_img_size', 0))

patch_step_size = int(config.get('patch_step_size', 0))

resize_height_width = int(config.get('resize_height_width', 0))


resize_img = config.get('resize_img')



resize_full_images = config.get('resize_full_images')


##################################################################################################################

import os
import logging
import warnings
import segmentation_models as sm

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_device
logging.getLogger('tensorflow').disabled = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')
os.environ["SM_FRAMEWORK"] = "tf.keras"

import time
import numpy as np
import cupy as cp
import random
import shutil
from itables import show

import tensorflow as tf
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import albumentations as A

from tensorflow import keras
import tensorflow.experimental.numpy as tnp
tnp.experimental_enable_numpy_behavior()
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, backend as K
from tensorflow.keras.initializers import HeUniform, he_normal
from tensorflow.keras.callbacks import Callback
from hist_metrics import *

from tensorflow.keras.layers import (
    Input, Activation, BatchNormalization, Conv2D, UpSampling2D, Add,
    GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, Multiply,
    Softmax, MaxPool2D, Concatenate, Dropout, DepthwiseConv2D, LeakyReLU
)

initializer = HeUniform(seed=None)


##################################################################################################################



path_1 = 'results/'
path_2 = 'logs/'
eval_results_path = 'eval_results/'

def create_and_remove_dirs():
    for i in range(5):
        if os.path.exists(path_1):
            shutil.rmtree(path_1)

        if os.path.exists(path_2):
            shutil.rmtree(path_2)
        
        if os.path.exists(eval_results_path):
            shutil.rmtree(eval_results_path)
            
    if not os.path.exists(path_1):
        os.makedirs(path_1)
        os.makedirs(path_2)
        os.makedirs(eval_results_path)
        

if (training):
    create_and_remove_dirs()


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    #tf.config.experimental.set_visible_devices(gpus[g_n], 'GPU')
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
    
    

print(tf. __version__)
# print(tf.keras. __version__)
    

seed = 13334
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)
tf.keras.utils.set_random_seed(seed)
os.environ['PYTHONHASHSEED']=str(seed)
cp.random.seed(seed=seed)





train_image_dir = train_images_patch_dir
train_mask_dir = train_masks_patch_dir


aug = A.Compose([
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        )
    ]
)


def binarize_mask(mask):
    # Convert mask values to binary (0 or 1)
    #mask = rgb2gray(mask)
    #mask = mask/255.
    # mask = (mask >= 0.5)
    #mask = preprocess_input(mask)
    #print('Min: %.3f, Max: %.3f' % (mask.min(), mask.max()))
    # print(np.unique(mask))
    mask = np.where(mask>0, 1, 0)
    # mask = mask.astype(np.uint8())
    #mask = aug(image=mask)['mask']
    #mask[mask == 255.0] = 1.0
    return mask

def normalize_image(image):
    #image = tf.keras.utils.normalize(image, axis=1)
    #image = preprocess_input(image)
    # print('Min: %.3f, Max: %.3f' % (image.min(), image.max()))
    # print(np.unique(image))
    #image = image / 255.
    image = aug(image=image)['image']
    return image



if (training):


    img_data_gen_args = dict( preprocessing_function = normalize_image , rotation_range=90, width_shift_range=0.3, height_shift_range=0.3, shear_range=0.5, zoom_range=0.3, horizontal_flip=True, vertical_flip=True, fill_mode='reflect')
    mask_data_gen_args = dict( preprocessing_function = binarize_mask , rotation_range=90, width_shift_range=0.3, height_shift_range=0.3, shear_range=0.5, zoom_range=0.3, horizontal_flip=True, vertical_flip=True, fill_mode='reflect')

    # img_data_gen_args = dict( preprocessing_function = normalize_image , rotation_range=90, horizontal_flip=True, vertical_flip=True, fill_mode='reflect')
    # mask_data_gen_args = dict( preprocessing_function = binarize_mask , rotation_range=90, horizontal_flip=True, vertical_flip=True, fill_mode='reflect')

    # img_data_gen_args = dict( preprocessing_function = normalize_image)
    # mask_data_gen_args = dict( preprocessing_function = binarize_mask )

    image_data_generator = ImageDataGenerator(**img_data_gen_args)
    mask_data_generator = ImageDataGenerator(**mask_data_gen_args)




    image_generator = image_data_generator.flow_from_directory(train_image_dir, 
                                                            seed=seed,target_size=(imgz_size, imgz_size), shuffle = True,
                                                            batch_size=batch_size,
                                                            class_mode=None)  #Very important to set this otherwise it returns multiple numpy arrays 
                                                                                #thinking class mode is binary.


    mask_generator = mask_data_generator.flow_from_directory(train_mask_dir, 
                                                            seed=seed,target_size=(imgz_size, imgz_size), shuffle = True,
                                                            batch_size=batch_size,
                                                            color_mode = 'grayscale',   #Read masks in grayscale
                                                            class_mode=None)





    train_generator = zip(image_generator, mask_generator)



    # x = image_generator.next()
    # y = mask_generator.next()
    # for i in range(0,1):
    #     image = x[i]
    #     mask = y[i]
    #     plt.subplot(1,2,1)
    #     plt.imshow(image)
    #     plt.subplot(1,2,2)
    #     plt.imshow(mask)
    #     plt.show()






class channel_attention_module(tf.keras.layers.Layer):
    def __init__(self, ratio=16):
        super(channel_attention_module, self).__init__()
        self.ratio = ratio

    def build(self, input_shape):
        self.channel = input_shape[-1]
        self.se_shape = (1, 1, self.channel)
        
        self.dense1 = Dense(self.channel // self.ratio, activation='swish', kernel_initializer=he_normal(), use_bias=False)
        self.dense2 = Dense(self.channel, activation='sigmoid', kernel_initializer=he_normal(), use_bias=False)

    def call(self, inputs):
        # Global Average Pooling
        se_avg = GlobalAveragePooling2D()(inputs)
        se_avg = Reshape(self.se_shape)(se_avg)
        
        # Global Max Pooling
        se_max = GlobalMaxPooling2D()(inputs)
        se_max = Reshape(self.se_shape)(se_max)
        
        # Combine both pooled features
        se = Add()([se_avg, se_max])
        
        # Pass through Dense layers
        se = self.dense1(se)
        se = self.dense2(se)
        
        # Scale the input features
        x = Multiply()([inputs, se])
        
        return x




def spatial_attention_module(x):
    x = tf.debugging.check_numerics(x, message='Input contains NaN or Inf')

    x_mean = tf.reduce_mean(x, axis=-1, keepdims=True)
    x_mean = tf.debugging.check_numerics(x_mean, message='x_mean contains NaN or Inf')


    x_max = tf.reduce_max(x, axis=-1, keepdims=True)
    x_max = tf.debugging.check_numerics(x_max, message='x_max contains NaN or Inf')


    x_min = tf.reduce_min(x, axis=-1, keepdims=True)
    x_min = tf.debugging.check_numerics(x_min, message='x_min contains NaN or Inf')


    x_sum = tf.reduce_sum(x, axis=-1, keepdims=True)
    x_sum = tf.debugging.check_numerics(x_sum, message='x_sum contains NaN or Inf')

    feats = Concatenate()([x_mean, x_max, x_min, x_sum])
    feats = tf.debugging.check_numerics(feats, message='feats contains NaN or Inf')
    
    feats = Dropout(0.5)(feats)
    

    feats = Conv2D(1, kernel_size=7, padding="same", kernel_initializer=he_normal(), activation=None)(feats)
    feats = Activation('swish')(feats)  # Using Swish activation before sigmoid
    feats = Conv2D(1, kernel_size=1, padding="same", kernel_initializer=he_normal(), activation="sigmoid")(feats)
    

    x = Multiply()([x, feats])
    x = tf.debugging.check_numerics(x, message='Output contains NaN or Inf')

    return x







def ca_sa(x):
    ca_x = channel_attention_module(ratio=16)(x)
    sa_x = spatial_attention_module(x)
    r = Add()([ca_x, sa_x])
    return r 



def conv_block(inputs, out_ch, rate=1, initializer=initializer):
    # Depthwise convolution
    x = DepthwiseConv2D(kernel_size=3, padding="same", dilation_rate=rate, use_bias=False, depthwise_initializer=initializer)(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    # Pointwise convolution
    x = Conv2D(out_ch, 1, padding="same", use_bias=False, kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    return x





def conv_block_res_path(x, num_filters, kernel_size, padding="same", act=True):
    # Depthwise Convolution
    x = DepthwiseConv2D(kernel_size, padding=padding, use_bias=False)(x)
    x = BatchNormalization()(x)
    if act:
        x = Activation("relu")(x)
    # Pointwise Convolution
    x = Conv2D(num_filters, (1, 1), padding=padding, use_bias=False)(x)
    x = BatchNormalization()(x)
    if act:
        x = Activation("relu")(x)
    return x

def res_path(x, num_filters, length):
    for i in range(length):
        x0 = x
        x1 = conv_block_res_path(x0, num_filters, 3, act=False)
        sc = conv_block_res_path(x0, num_filters, 1, act=False)
        x = Add()([x1, sc])
        x = LeakyReLU()(x)
        x = BatchNormalization()(x)
    return x


def linear_attention(x, in_channels):
    keys = Dense(in_channels)(x)
    queries = Dense(in_channels)(x)
    values = Dense(in_channels)(x)
    softmax = Softmax(axis=-1)
    
    attention_weights = softmax(tf.matmul(queries, keys, transpose_b=True))
    out = tf.matmul(attention_weights, values)
    return out






layer_count_attn = 1


# Attention structure
FILTER_NUM = 64 # number of basic filters for the first layer
FILTER_SIZE = 3 # size of the convolutional filter
UP_SAMP_SIZE = 2 # size of upsampling filters

def repeat_elem(tensor, rep):

    return layers.Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3),
                        arguments={'repnum': rep})(tensor)

def gating_signal(input, out_size):

    x = layers.Conv2D(out_size, (1, 1), padding='same')(input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def attention_block(x, gating, inter_shape):
    shape_x = K.int_shape(x)
    shape_g = K.int_shape(gating)


    theta_x = layers.Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(x)  # 16
    shape_theta_x = K.int_shape(theta_x)


    phi_g = layers.Conv2D(inter_shape, (1, 1), padding='same')(gating)
    upsample_g = layers.Conv2DTranspose(inter_shape, (3, 3),
                                strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),
                                padding='same')(phi_g)  # 16

    concat_xg = layers.add([upsample_g, theta_x])
    act_xg = layers.Activation('relu')(concat_xg)
    psi = layers.Conv2D(1, (1, 1), padding='same')(act_xg)
    sigmoid_xg = layers.Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    upsample_psi = layers.UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32

    upsample_psi = repeat_elem(upsample_psi, shape_x[3])

    y = layers.multiply([upsample_psi, x])

    global layer_count_attn
    result = layers.Conv2D(shape_x[3], (1, 1), padding='same' )(y)
    
    result_bn = layers.BatchNormalization(name='attention_'+str(layer_count_attn))(result)
    layer_count_attn = layer_count_attn + 1
    return result_bn


def I_UNet_7(inputs, out_ch, int_ch, num_layers, rate=2):
    """ Initial Conv """
    x = conv_block(inputs, out_ch)
    init_feats = x
    

    """ Encoder """
    skip = []
    x = conv_block(x, int_ch)  
    skip.append(x)

    x = MaxPool2D((2, 2))(x)
    x_128 = conv_block(x, int_ch)
    skip.append(x_128)
    
    x = MaxPool2D((2, 2))(x_128)
    x_64 = conv_block(x, int_ch)
    skip.append(x_64)
    
    x = MaxPool2D((2, 2))(x_64)
    x_32 = conv_block(x, int_ch)
    skip.append(x_32)
    
    x = MaxPool2D((2, 2))(x_32)
    x_16 = conv_block(x, int_ch)
    skip.append(x_16)
    
    x = MaxPool2D((2, 2))(x_16)
    x = conv_block(x, int_ch)
    skip.append(x)



    """ Bridge """
    x = conv_block(x, int_ch, rate=rate)
    
    
    
    """ Decoder """
    skip.reverse()

    x = Concatenate()([x, skip[0]])
    x = conv_block(x, int_ch)
    
    #-------------------------------------------------------------#
    gating_16 = gating_signal(x, 8*FILTER_NUM)
    att_16 = attention_block(x_16, gating_16, 8*FILTER_NUM)
    x = UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
    x = layers.concatenate([x, att_16], axis=3)
    x = conv_block(x, int_ch)
    #-------------------------------------------------------------#
    
    
    #-------------------------------------------------------------#
    gating_32 = gating_signal(x, 4*FILTER_NUM)
    att_32 = attention_block(x_32, gating_32, 4*FILTER_NUM)
    x = UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
    x = layers.concatenate([x, att_32], axis=3)
    x = conv_block(x, int_ch)
    #-------------------------------------------------------------#
    

    #-------------------------------------------------------------#
    gating_64 = gating_signal(x, 2*FILTER_NUM)
    att_64 = attention_block(x_64, gating_64, 2*FILTER_NUM)
    x = UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
    x = layers.concatenate([x, att_64], axis=3)
    x = conv_block(x, int_ch)
    #-------------------------------------------------------------#
    
    
    #-------------------------------------------------------------#
    gating_128 = gating_signal(x, FILTER_NUM)
    att_128 = attention_block(x_128, gating_128, FILTER_NUM)
    x = UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
    x = layers.concatenate([x, att_128], axis=3)
    x = conv_block(x, int_ch)
    #-------------------------------------------------------------#


    x = UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
    x = Concatenate()([x, skip[-1]])
    x = conv_block(x, out_ch)

    """ Add """
    x = Add()([x, init_feats])
    return x


def I_UNet_6(inputs, out_ch, int_ch, num_layers, rate=2):
    """ Initial Conv """
    x = conv_block(inputs, out_ch)
    init_feats = x
    

    """ Encoder """
    skip = []
    x = conv_block(x, int_ch)
    #print(x.shape)
    skip.append(x)
    


    x = MaxPool2D((2, 2))(x)
    x_64 = conv_block(x, int_ch)
    #print(x_64.shape)
    skip.append(x_64)
    
    x = MaxPool2D((2, 2))(x_64)
    x_32 = conv_block(x, int_ch)
    #print(x_32.shape)
    skip.append(x_32)
    
    x = MaxPool2D((2, 2))(x_32)
    x_16 = conv_block(x, int_ch)
    #print(x_16.shape)
    skip.append(x_16)
    
    x = MaxPool2D((2, 2))(x_16)
    x = conv_block(x, int_ch)
    #print(x.shape)
    skip.append(x)
    



    """ Bridge """
    x = conv_block(x, int_ch, rate=rate)  # (None, 8, 8, 32)
    
    
    
    """ Decoder """
    skip.reverse()

    x = Concatenate()([x, skip[0]])
    x = conv_block(x, int_ch)


    #-------------------------------------------------------------#
    gating_16 = gating_signal(x, 8*FILTER_NUM)
    att_16 = attention_block(x_16, gating_16, 8*FILTER_NUM)
    x = UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
    x = layers.concatenate([x, att_16], axis=3)
    x = conv_block(x, int_ch)
    #-------------------------------------------------------------#
    
    
    #-------------------------------------------------------------#
    gating_32 = gating_signal(x, 4*FILTER_NUM)
    att_32 = attention_block(x_32, gating_32, 4*FILTER_NUM)
    x = UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
    x = layers.concatenate([x, att_32], axis=3)
    x = conv_block(x, int_ch)
    #-------------------------------------------------------------#
    

    #-------------------------------------------------------------#
    gating_64 = gating_signal(x, 2*FILTER_NUM)
    att_64 = attention_block(x_64, gating_64, 2*FILTER_NUM)
    x = UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
    x = layers.concatenate([x, att_64], axis=3)
    x = conv_block(x, int_ch)
    #-------------------------------------------------------------#




    x = UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
    x = Concatenate()([x, skip[-1]])
    x = conv_block(x, out_ch)

    """ Add """
    x = Add()([x, init_feats])
    return x



def I_UNet_5(inputs, out_ch, int_ch, num_layers, rate=2):
    """ Initial Conv """
    x = conv_block(inputs, out_ch)
    init_feats = x
    

    """ Encoder """
    skip = []
    x = conv_block(x, int_ch)
    skip.append(x)

    
    x = MaxPool2D((2, 2))(x)
    x_32 = conv_block(x, int_ch)
    skip.append(x_32)
    
    x = MaxPool2D((2, 2))(x_32)
    x_16 = conv_block(x, int_ch)
    skip.append(x_16)
    
    x = MaxPool2D((2, 2))(x_16)
    x = conv_block(x, int_ch)
    skip.append(x)
    



    """ Bridge """
    x = conv_block(x, int_ch, rate=rate)
    
    
    
    """ Decoder """
    skip.reverse()

    x = Concatenate()([x, skip[0]])
    x = conv_block(x, int_ch)


    #-------------------------------------------------------------#
    gating_16 = gating_signal(x, 8*FILTER_NUM)
    att_16 = attention_block(x_16, gating_16, 8*FILTER_NUM)
    x = UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
    x = layers.concatenate([x, att_16], axis=3)
    x = conv_block(x, int_ch)
    #-------------------------------------------------------------#
    
    
    #-------------------------------------------------------------#
    gating_32 = gating_signal(x, 4*FILTER_NUM)
    att_32 = attention_block(x_32, gating_32, 4*FILTER_NUM)
    x = UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
    x = layers.concatenate([x, att_32], axis=3)
    x = conv_block(x, int_ch)
    #-------------------------------------------------------------#
    

    x = UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
    x = Concatenate()([x, skip[-1]])
    x = conv_block(x, out_ch)

    """ Add """
    x = Add()([x, init_feats])
    return x



def I_UNet_4(inputs, out_ch, int_ch, num_layers, rate=2):
    """ Initial Conv """
    x = conv_block(inputs, out_ch)
    init_feats = x
    

    """ Encoder """
    skip = []
    x = conv_block(x, int_ch)
    skip.append(x)


    
    x = MaxPool2D((2, 2))(x)
    x_16 = conv_block(x, int_ch)
    skip.append(x_16)
    
    x = MaxPool2D((2, 2))(x_16)
    x = conv_block(x, int_ch)
    skip.append(x)
    



    """ Bridge """
    x = conv_block(x, int_ch, rate=rate)
    
    
    
    """ Decoder """
    skip.reverse()

    x = Concatenate()([x, skip[0]])
    x = conv_block(x, int_ch)


    #-------------------------------------------------------------#
    gating_16 = gating_signal(x, 8*FILTER_NUM)
    att_16 = attention_block(x_16, gating_16, 8*FILTER_NUM)
    x = UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
    x = layers.concatenate([x, att_16], axis=3)
    x = conv_block(x, int_ch)
    #-------------------------------------------------------------#
    
    

    x = UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
    x = Concatenate()([x, skip[-1]])
    x = conv_block(x, out_ch)

    """ Add """
    x = Add()([x, init_feats])

    return x



def bridge_block(inputs, out_ch, int_ch):
    """ Initial Conv """
    x0 = conv_block(inputs, out_ch, rate=1)

    """ Encoder """
    x1 = conv_block(x0, int_ch, rate=1)
    x2 = conv_block(x1, int_ch, rate=2)
    x3 = conv_block(x2, int_ch, rate=4)

    """ Bridge """
    x4 = conv_block(x3, int_ch, rate=8)

    """ Decoder """
    x = Concatenate()([x4, x3])
    x = conv_block(x, int_ch, rate=4)

    x = Concatenate()([x, x2])
    x = conv_block(x, int_ch, rate=2)

    x = Concatenate()([x, x1])
    x = conv_block(x, out_ch, rate=1)

    """ Addition """
    x = Add()([x, x0])
    return x

def mcads(input_shape, out_ch, int_ch, num_classes=2):
    """ Input Layer """
    inputs = Input(input_shape)
    s0 = inputs

    """ Encoder """
    
    s1 = I_UNet_7(s0, out_ch[0], int_ch[0], 7)
    s1 = ca_sa(s1)
    s1 = conv_block(s1, out_ch[0])
    p1 = MaxPool2D((2, 2))(s1)

    s2 = I_UNet_6(p1, out_ch[1], int_ch[1], 6)
    s2 = ca_sa(s2)
    s2 = conv_block(s2, out_ch[1])
    p2 = MaxPool2D((2, 2))(s2)

    s3 = I_UNet_5(p2, out_ch[2], int_ch[2], 5)
    s3 = ca_sa(s3)
    s3 = conv_block(s3, out_ch[2])
    p3 = MaxPool2D((2, 2))(s3)

    s4 = I_UNet_4(p3, out_ch[3], int_ch[3], 4)
    s4 = ca_sa(s4)
    s4 = conv_block(s4, out_ch[3])
    p4 = MaxPool2D((2, 2))(s4)

    s5 = bridge_block(p4, out_ch[4], int_ch[4])
    s5 = ca_sa(s5)
    s5 = conv_block(s5, out_ch[4])
    p5 = MaxPool2D((2, 2))(s5)
    


    """ Bridge """
    b1 = bridge_block(p5, 512, int_ch[5])

    
    """ Decoder """
    
    downsample_factor = 2
    
    #------#
    output_channels = b1.shape[-1]
    num_filters = output_channels * (downsample_factor ** 2)
    u1 = Conv2D(num_filters, 3, padding="same", activation="relu", kernel_initializer=initializer)(b1)
    u1 = tf.nn.depth_to_space(u1, downsample_factor)
    u1 = Conv2D(u1.shape[-1], 3, padding="same", activation="relu", kernel_initializer=initializer)(u1)
    # print("u1 shape: ", u1.shape)
    u1 = conv_block(u1, s5.shape[-1])
    
    att_5 = res_path(s5, s5.shape[-1], 5)
    d1 = Concatenate()([u1, att_5])
    d1 = conv_block(d1, s5.shape[-1])
    d1 = linear_attention(d1, d1.shape[-1])
    d1 = Add()([u1, d1])
    d1 = conv_block(d1, d1.shape[-1])
    d1 = ca_sa(d1)
    # print("d1 shape: ", d1.shape)
    #------#
    
    #------#
    output_channels = d1.shape[-1]
    num_filters = output_channels * (downsample_factor ** 2)
    u2 = Conv2D(num_filters, 3, padding="same", activation="relu", kernel_initializer=initializer)(d1)
    u2 = tf.nn.depth_to_space(u2, downsample_factor)
    u2 = Conv2D(u2.shape[-1], 3, padding="same", activation="relu", kernel_initializer=initializer)(u2)
    # print("u2 shape: ", u2.shape)
    u2 = conv_block(u2, s4.shape[-1])
    
    att_4 = res_path(s4, s4.shape[-1], 4)
    d2 = Concatenate()([u2, att_4])
    d2 = conv_block(d2, s4.shape[-1])
    d2 = linear_attention(d2, d2.shape[-1])
    d2 = Add()([u2, d2])
    d2 = conv_block(d2, d2.shape[-1])
    d2 = ca_sa(d2)
    # print("d2 shape: ", d2.shape)
    #------#
    
    u3 = UpSampling2D(size=(2, 2), interpolation="bilinear")(d2)
    u3 = conv_block(u3, s3.shape[-1])
    # print("u3 shape: ", u3.shape)
    att_3 = res_path(s3, s3.shape[-1], 3)
    d3 = Concatenate()([u3, att_3])
    d3 = conv_block(d3, s3.shape[-1])
    d3 = linear_attention(d3, d3.shape[-1])
    d3 = Add()([u3, d3])
    d3 = conv_block(d3, d3.shape[-1])
    d3 = ca_sa(d3)
    # print("d3 shape: ", d3.shape)
    
    
    u4 = UpSampling2D(size=(2, 2), interpolation="bilinear")(d3)
    u4 = conv_block(u4, s2.shape[-1])
    # print("u4 shape: ", u4.shape)
    att_2 = res_path(s2, s2.shape[-1], 2)
    att_2 = linear_attention(att_2, att_2.shape[-1])
    d4 = Concatenate()([u4, att_2])
    d4 = conv_block(d4, s2.shape[-1])
    d4 = linear_attention(d4, d4.shape[-1])
    d4 = Add()([u4, d4])
    d4 = conv_block(d4, d4.shape[-1])
    d4 = ca_sa(d4)
    # print("d4 shape: ", d4.shape)


    u5 = UpSampling2D(size=(2, 2), interpolation="bilinear")(d4)
    u5 = conv_block(u5, s1.shape[-1])
    # print("u5 shape: ", u5.shape)
    att_1 = res_path(s1, s1.shape[-1], 1)
    d5 = Concatenate()([u5, att_1])
    d5 = conv_block(d5, s1.shape[-1])
    d5 = linear_attention(d5, d5.shape[-1])
    d5 = Add()([u5, d5])
    d5 = conv_block(d5, d5.shape[-1])
    d5 = ca_sa(d5)
    # print("d5 shape: ", d5.shape)

    





    """ Side Outputs """
    

    z1 = Conv2D(num_classes, 3, padding="same", use_bias=False, kernel_initializer = initializer)(d5)
    
    z2 = UpSampling2D(size=(2, 2), interpolation="bilinear")(d4)
    z2 = Conv2D(num_classes, 3, padding="same", use_bias=False, kernel_initializer = initializer)(z2)
    
    z3 = UpSampling2D(size=(4, 4), interpolation="bilinear")(d3)
    z3 = Conv2D(num_classes, 3, padding="same", use_bias=False, kernel_initializer = initializer)(z3)
    
    z4 = UpSampling2D(size=(8, 8), interpolation="bilinear")(d2)
    z4 = Conv2D(num_classes, 3, padding="same", use_bias=False, kernel_initializer = initializer)(z4)
    
    z5 = UpSampling2D(size=(16, 16), interpolation="bilinear")(d1)
    z5 = Conv2D(num_classes, 3, padding="same", use_bias=False, kernel_initializer = initializer)(z5)
    
    z6 = UpSampling2D(size=(32, 32), interpolation="bilinear")(b1)
    z6 = Conv2D(num_classes, 3, padding="same", use_bias=False, kernel_initializer = initializer)(z6)

    
    z1_act = Activation("sigmoid")(z1)
    z2_act = Activation("sigmoid")(z2)
    z3_act = Activation("sigmoid")(z3)
    z4_act = Activation("sigmoid")(z4)
    z5_act = Activation("sigmoid")(z5)
    z6_act = Activation("sigmoid")(z6)

    model = tf.keras.models.Model(inputs, outputs=[z6_act, z5_act, z4_act, z3_act, z2_act, z1_act])
    return model

def build_model(input_shape, num_classes=1):
    out_ch = [64, 128, 256, 512, 512, 512, 512, 256, 128, 64, 64]
    int_ch = [32, 32, 64, 128, 256, 256, 256, 128, 64, 32, 16]
    model = mcads(input_shape, out_ch, int_ch, num_classes=num_classes)
    return model


def build_model_mcads():
    return build_model((256, 256, 3))



opt = tf.keras.optimizers.legacy.Adam(learning_rate=1e-4)
BCE = keras.losses.BinaryCrossentropy(from_logits=False)
iou_score = sm.metrics.IOUScore(per_image = True, threshold = 0.5)



model = build_model_mcads()
model.compile( optimizer=opt, loss = BCE , metrics=[iou_score] )

if (training):

    class TimeHistory(Callback):
        def on_train_begin(self, logs={}):
            self.times = []

        def on_epoch_begin(self, epoch, logs={}):
            self.epoch_start_time = time.time()

        def on_epoch_end(self, epoch, logs={}):
            self.times.append(time.time() - self.epoch_start_time)
            # print(f"Epoch {epoch+1} took {self.times[-1]:.2f} seconds")
            print(f"Epoch {epoch+1} took {format_time(self.times[-1])}")

    time_callback = TimeHistory()


    filepath= path_1 + "{epoch:04d}.hdf5"
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=filepath, save_best_only=False, save_weights_only=True, verbose= 1)


    csv_logger = tf.keras.callbacks.CSVLogger(path_2 + "logs.csv", separator=",", append=False)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.01, patience=5, min_lr=1e-16, verbose=1)


    num_train_imgs = len(os.listdir(train_image_dir + 'train/'))
    steps_per_epoch = num_train_imgs //batch_size
    print("num_train_imgs -------------- = " + str(num_train_imgs))
    print("steps_per_epoch -------------- = " + str(steps_per_epoch))



    callbacks_list = [checkpoint, csv_logger, time_callback, reduce_lr]


# %%
if (training):
    start_time = time.time()

    history = model.fit(train_generator,
                        steps_per_epoch=steps_per_epoch,
                        epochs=num_epochs,
                        callbacks=callbacks_list,
                        batch_size=batch_size, verbose=1)
 


    end_time = time.time()

    total_training_time = end_time - start_time
    total_training_time_formatted = format_time(total_training_time)

    print(f"Total training time: {total_training_time:.3f} seconds")
    print(f"Total training time: {total_training_time_formatted}")

    model.save_weights(path_1 + 'last_weights.hdf5')
    
    epoch_times = time_callback.times
    calc_time(epoch_times)







####---------------------------------------------------------Evaluation Evaluation Evaluation Evaluation Evaluation Evaluation Evaluation Evaluation Evaluation Evaluation Evaluation---------------------------------------------------------###

print("####------------Testing start------------####")


test_images_arg = test_images_patch_dir
test_masks_arg = test_masks_patch_dir



X_test = np.load(test_images_arg)
y_test = np.load(test_masks_arg)


print(X_test.shape)
print(y_test.shape)


X_test = aug(image=X_test)['image']
print('Min: %.3f, Max: %.3f' % (X_test.min(), X_test.max()))
# print(np.unique(X_test))

y_test = np.where(y_test>0, 1, 0)
y_test = y_test.astype(np.float32())
print('Min: %.3f, Max: %.3f' % (y_test.min(), y_test.max()))
print(np.unique(y_test))

# %%
# i = random.randint(0, X_test.shape[0]-1)
# print(i)
# image = X_test[i]
# mask = y_test[i]
# plt.subplot(1,2,1)
# plt.imshow(image)
# plt.subplot(1,2,2)
# plt.imshow(mask)
# plt.show()

def model_clear_and_build():
    tf.keras.backend.clear_session()

del model

model_clear_and_build()

model = build_model_mcads()

model_clear_and_build()

pred_index = -1
df_combined = 0
df_counter = 0

current_cwd = os.getcwd()
weights_folder_path = current_cwd + "/results/"
dirs_ = sorted( os.listdir(weights_folder_path) )

try:
    dirs_.remove('last_weights.hdf5')
    dirs_.remove('last_weights_2.hdf5')
    dirs_.remove('last_weights_3.hdf5')
except:
    pass


pred = model.predict(X_test[0:3], batch_size=1, verbose=0)

if isinstance(pred, list):
    len_pred = len(pred)
else:
    len_pred = 1

print("No of Prediction Activations: ", len_pred)

if len_pred == X_test.shape[0]:
    len_pred = 1
    pred_index = -1
elif (len_pred == 1):
    len_pred = 1
    pred_index = -1
else:
    pred_index = 0


print("Pred_Index: ", pred_index)






#### Controling Predictions

# 0 = all predictions
# 1 = specific prediction
# 2 = default


control_sup = 1

if (control_sup == 1):

    ##################
    len_pred = 1
    pred_index = 5
    print("No of Prediction Activations: ", len_pred)
    print("Pred_Index: ", pred_index)
    ##################



# %%
if (training):
    def df_convert(df, ep_str, index_str):
        df = df.transpose()
        #df = df[1:]
        df.columns = df.iloc[0]
        df = df[1:]
        df.reset_index(drop=True, inplace=True)
        df.insert(0, 'Index_Out', index_str)
        df.insert(0, 'Epoch', ep_str)
        return df

    for f in range(0, len(dirs_)):
        pbar = tqdm( range(0, len_pred) )
        epoch_f = dirs_[f]

        for p in pbar:
            
            model_clear_and_build()
            
            pbar.set_description("Epoch: " + epoch_f)
            model.load_weights(weights_folder_path + dirs_[f])
            #print(dirs_[f])
            
            pred_sm_f = model.predict(X_test, verbose=0, batch_size=1)
    
            if (control_sup == 0):
                pred_index = p
                pred_sm_f = pred_sm_f[pred_index]
            
            if (control_sup == 1):
                pred_index = pred_index
                pred_sm_f = pred_sm_f[pred_index]
            

            
            df = eval_all_metrics(y_test, pred_sm_f)
            df = df.round(4)
            
            df = df_convert(df, epoch_f, pred_index)
            
            if df_counter == 0:
                df_combined = df
                
                df_counter = 1
            else:
                df_combined = pd.concat([df_combined, df], ignore_index=True)
            
            df_combined.to_csv(eval_results_path +"Epoch_"+ epoch_f +"_eval_all_metrics.csv")
            #print(tabulate(df_combined, headers = 'keys', tablefmt = 'psql'))
                
        
            
    df_combined.to_csv(eval_results_path + "all_eval.csv")

# %%
if (training):
    df = pd.read_csv(eval_results_path + "all_eval.csv")

    # Find the rows with maximum values in "sm_iou_score_per_image" grouped by "Epoch" and "Index_Out"
    max_rows = df.loc[df.groupby([ "Index_Out"])["sm_iou_score_per_image"].idxmax()]
    max_rows = max_rows.drop(columns=['Unnamed: 0'])
    max_rows.to_csv(eval_results_path + "all_eval_max_filter.csv")
    show(max_rows)

# %%
df = pd.read_csv(eval_results_path + "all_eval_max_filter.csv")
max_iou_index = df['sm_iou_score_per_image'].idxmax()
max_iou_row = df.iloc[max_iou_index]
max_iou_weight_file = weights_folder_path + max_iou_row[1]
best_epoch_str = max_iou_row[1]
print(max_iou_weight_file)
print(best_epoch_str)
pred_index = df.iloc[max_iou_index][2]
print(pred_index)

# %%
#####################################################################################################################################################################

# %%
model_clear_and_build()
model.load_weights(max_iou_weight_file)

# %%
threshold_pred = 0.5

# %%
#####################################################################################################################################################################

# %%
style_df_list = eval_tta_patches(model, X_test, y_test, pred_index, threshold_pred, training, eval_results_path, best_epoch_str, model_clear_and_build, path_1)

# %%
# all_metrics_
style_df_list[0]

# %%
# all_metrics_tta_
style_df_list[1]

# %%
style_df_list = eval_tta_swa_patches(model, X_test, y_test, pred_index, threshold_pred, training, eval_results_path, best_epoch_str, model_clear_and_build, path_1, weights_folder_path = "results/")

# %%
# all_metrics_avgW_
style_df_list[0]

# %%
# all_metrics_tta_avgW_
style_df_list[1]

# %%
##############################----------------------------------------------------------------------------------------------------------##############################
##############################----------------------------------------------------------------------------------------------------------##############################
##############################----------------------------------------------------------------------------------------------------------##############################
##############################----------------------------------------------------------------------------------------------------------##############################
##############################----------------------------------------------------------------------------------------------------------##############################

# %%
model_clear_and_build()
model.load_weights(max_iou_weight_file)


create_pred_dirs()



resize_hw = [resize_height_width, resize_height_width]


# pred
# pred_tta
# pred_avg
# pred_avg_tta

# %%
style_df_list = evaluation_full(control_sup, pred_index, model, image_full_test_directory, mask_full_test_directory, patch_img_size, patch_step_size, resize_img, resize_hw, resize_full_images, eval_results_path, threshold_pred, model_clear_and_build)

# %%
# all_metrics_full_pred
style_df_list[0]

# %%
# all_metrics_full_pred_05
style_df_list[1]

# %%
# all_metrics_full_pred_tta
style_df_list[2]

# %%
# all_metrics_full_pred_05_tta
style_df_list[3]

# %%
style_df_list = evaluation_full_Swa(control_sup, pred_index, model, image_full_test_directory, mask_full_test_directory, patch_img_size, patch_step_size, resize_img, resize_hw, resize_full_images, eval_results_path, threshold_pred, model_clear_and_build)

# %%
# all_metrics_full_pred_avgW
style_df_list[0]

# %%
# all_metrics_full_pred_05_avgW
style_df_list[1]

# %%
# all_metrics_full_pred_tta_avgW
style_df_list[2]

# %%
# all_metrics_full_pred_05_tta_avgW
style_df_list[3]

# %%
style_df_list = evaluation_full_last_epochs(control_sup, pred_index, model, image_full_test_directory, mask_full_test_directory, patch_img_size, patch_step_size, resize_img, resize_hw, resize_full_images, eval_results_path, threshold_pred, model_clear_and_build, path_1, weights_folder_path = "results/")

# %%
# all_metrics_full_pred_avgW_last_epochs
style_df_list[0]

# %%
# all_metrics_full_pred_05_avgW_last_epochs
style_df_list[1]

# %%
# all_metrics_full_pred_tta_avgW_last_epochs
style_df_list[2]

# %%
# all_metrics_full_pred_05_tta_avgW_last_epochs
style_df_list[3]


