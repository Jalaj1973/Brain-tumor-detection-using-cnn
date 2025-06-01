import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model=load_model('BrainTumor10EpochsCategorical.h5')

image=cv2.imread('/Users/jalajbalodi/Desktop/BrainTumor Classification DL/uploads/pred5.jpg')

img=Image.fromarray(image)

img=img.resize((64,64))

img=np.array(img)

input_img=np.expand_dims(img, axis=0)

result = np.argmax(model.predict(input_img), axis=-1)
print(result)

if result == 0:
    print("No Brain Tumor")
elif result == 1:
    print("Yes Brain Tumor")





