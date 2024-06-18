import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import cv2 as cv
import os
import random
import rawpy
import requests
from tensorflow.keras.utils import array_to_img, img_to_array,load_img
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D,concatenate,Cropping2D,Conv2DTranspose,BatchNormalization,Dropout
from keras.optimizers import Adam,RMSprop
from tensorflow.keras.callbacks import EarlyStopping,      ReduceLROnPlateau
from PIL import Image, UnidentifiedImageError
from skimage.metrics import peak_signal_noise_ratio as psnr, mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

batch_size=32

img_size=256
epochs=100


def load_images(img_fold, image_size=(img_size, img_size)):
    images = []
    for img_name in sorted(os.listdir(img_fold)):
        img_path=os.path.join(img_fold, img_name)
        
        img = load_img(img_path, target_size=image_size)
        img = img_to_array(img)/255 
        images.append(img)
        # augmentation x more samples
        img1=cv.flip(img,1)
        images.append(img_to_array(img1))
        img2=cv.flip(img,-1)
        images.append(img_to_array(img2))
        img3=cv.rotate(img,cv.ROTATE_90_CLOCKWISE)
        images.append(img_to_array(img3))
        img4=cv.rotate(img,cv.ROTATE_90_COUNTERCLOCKWISE)
        images.append(img_to_array(img4))
            
       
    return np.array(images)


def unet(input_shape=(img_size,img_size,3)):
    inputs=Input(input_shape)
    
    #encoder part
    dlayer1 = layers.Conv2D(128, (3, 3), padding='same', strides=2)(inputs)
    dlayer1 = layers.BatchNormalization()(dlayer1)
    dlayer1 = layers.LeakyReLU()(dlayer1)
    
    dlayer2 = layers.Conv2D(128, (3, 3), padding='same', strides=2)(dlayer1)
    dlayer2 = layers.BatchNormalization()(dlayer2)
    dlayer2 = layers.LeakyReLU()(dlayer2)
    
    dlayer3 = layers.Conv2D(256, (3, 3), padding='same', strides=2)(dlayer2)
    dlayer3 = layers.BatchNormalization()(dlayer3)
    dlayer3 = layers.LeakyReLU()(dlayer3)
    
    dlayer4 = layers.Conv2D(512, (3, 3), padding='same', strides=2)(dlayer3)
    dlayer4 = layers.BatchNormalization()(dlayer4)
    dlayer4 = layers.LeakyReLU()(dlayer4)
    
    dlayer5 = layers.Conv2D(512, (3, 3), padding='same', strides=2)(dlayer4)
    dlayer5 = layers.BatchNormalization()(dlayer5)
    dlayer5 = layers.LeakyReLU()(dlayer5)




    ulayer1 = layers.Conv2DTranspose(512, (3, 3), padding='same', strides=2)(dlayer5)
    ulayer1 = layers.Dropout(0.09)(ulayer1)
    ulayer1 = layers.LeakyReLU()(ulayer1)
    ulayer1 = layers.concatenate([ulayer1, dlayer4])
   
    
    ulayer2 = layers.Conv2DTranspose(256, (3, 3), padding='same', strides=2)(ulayer1)
    ulayer2 = layers.Dropout(0.09)(ulayer2)
    ulayer2 = layers.LeakyReLU()(ulayer2)
    ulayer2 = layers.concatenate([ulayer2, dlayer3])
    
    ulayer3 = layers.Conv2DTranspose(128, (3, 3), padding='same', strides=2)(ulayer2)
    ulayer3 = layers.Dropout(0.09)(ulayer3)
    ulayer3 = layers.LeakyReLU()(ulayer3)
    ulayer3 = layers.concatenate([ulayer3, dlayer2])
    
    ulayer4 = layers.Conv2DTranspose(128, (3, 3), padding='same', strides=2)(ulayer3)
    ulayer4 = layers.Dropout(0.09)(ulayer4)
    ulayer4 = layers.LeakyReLU()(ulayer4)
    ulayer4 = layers.concatenate([ulayer4, dlayer1])
    
    ulayer5 = layers.Conv2DTranspose(3, (3, 3), padding='same', strides=2)(ulayer4)
    ulayer5 = layers.Dropout(0.09)(ulayer5)
    ulayer5 = layers.LeakyReLU()(ulayer5)
    ulayer5 = layers.concatenate([ulayer5, inputs])
    
    output = layers.Conv2D(3, (2, 2), strides=1, padding='same')(ulayer5)
    return Model(inputs=inputs, outputs=output)


def train_model():
    clean_images=load_images('./Train/high')
    noisy_images=load_images('./Train/low')

    #using unet
    model=unet(input_shape=(img_size,img_size,3))        
    model.compile(optimizer=RMSprop(learning_rate=0.001),loss='mean_absolute_error',metrics=['accuracy'])
    
    early_stopping=EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)
    lr_reduction=ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=5,min_lr=0.00001)

    model.fit(noisy_images,clean_images,epochs=epochs,batch_size=batch_size,verbose=1,validation_data=(noisy_images,clean_images),callbacks=[early_stopping,lr_reduction])

    return model


def plot_images(low, high, predicted):
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.title("low")
    plt.imshow(low)

    plt.subplot(1, 3, 2)
    plt.title("high ")
    plt.imshow(high)

    plt.subplot(1, 3, 3)
    plt.title("predicted")
    plt.imshow(predicted)

    plt.show()


def evaluate(model, clean_images, noisy_images):
    predictions = model.predict(noisy_images)
    mse_score = [mse(clean, pred) for clean, pred in zip(clean_images, predictions)]
    psnr_score = [psnr(clean, pred) for clean, pred in zip(clean_images, predictions)]
    mae_score = [mae(clean.flatten(), pred.flatten()) for clean, pred in zip(clean_images, predictions)]

    print(f"Mean MSE: {np.mean(mse_score)}")
    print(f"Mean PSNR: {np.mean(psnr_score)}")
    print(f"Mean MAE: {np.mean(mae_score)}")

if __name__ == '__main__':
    model =train_model()
    
    clean_images=load_images('./Train/high')
    noisy_images=load_images('./Train/low')
    
    evaluate(model, clean_images, noisy_images)
    
    for i   in range (0, 25, 5):
        predicted=np.clip(model.predict(noisy_images[i].reshape(1  , img_size, img_size, 3)), 0.0, 1.0).reshape(img_size, img_size, 3)
        plot_images(noisy_images[i], clean_images[i], predicted)
    
evaluate(model, clean_images, noisy_images)



test_low_images = './test/low'
output_images = './test/predicted'

if not os.path.exists(output_images):
    os.makedirs(output_images)


def preprocessImage(img_path, targetSize=(256, 256)):
    img = load_img(img_path, target_size=targetSize)
    imgArray = img_to_array(img)
    imgArray = np.expand_dims(imgArray, axis=0)
    imgArray /= 255.0
    return imgArray

def savePredictionImage(prediction, outputPath):
    prediction = np.squeeze(prediction, axis=0)
    prediction = np.clip(prediction * 255.0, 0, 255).astype('uint8')
    img = array_to_img(prediction)
    img.save(outputPath)

def predictImage(model, img_path):
    img = preprocessImage(img_path)
    prediction = model.predict(img)
    return prediction

for img_name in os.listdir(test_low_images):
    img_path = os.path.join(test_low_images, img_name)
    if os.path.isfile(img_path) and img_name.lower().endswith(('png', 'jpg', 'jpeg')):
        prediction = predictImage(model, img_path)
        print(f'Prediction for {img_name}: {prediction.shape}')
        outputPath = os.path.join(output_images, img_name)
        savePredictionImage(prediction, outputPath)