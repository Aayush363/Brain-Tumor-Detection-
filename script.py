import numpy as np 
import tensorflow as tf 
from tensorflow import keras
from keras.models import Model
from keras.layers import Flatten, Dense, Dropout
from keras.applications.vgg16 import VGG16
import cv2
import os
import sys
import imutils

model = VGG16(include_top = False, input_shape = (224, 224) + (3, ))
for layer in model.layers:
    layer.trainable = False

flat1 = Flatten()(model.layers[-1].output)
class1 = Dense(1024, activation='relu')(flat1)
drop = Dropout(0.5)(class1)
class2 = Dense(512, activation='relu')(drop)
class3 = Dense(256, activation='relu')(class2)
class4 = Dense(128, activation='relu')(class3)
output = Dense(2, activation='softmax')(class4)

model = Model(inputs=model.inputs, outputs=output)
dirname = os.path.dirname(__file__)
weights = os.path.join(dirname, "mdl_wts2.hdf5")
model.load_weights(weights)

img_path = os.path.join(dirname, "public/uploaded_images/" + str(sys.argv[1]))
img = cv2.imread(img_path)

IMG_SIZE = (224,224)
img = cv2.resize(
            img,
            dsize=IMG_SIZE,
            interpolation=cv2.INTER_CUBIC
        )
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)

# threshold the image, then perform a series of erosions +
# dilations to remove any small regions of noise
thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=2)

# find contours in thresholded image, then grab the largest one
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
c = max(cnts, key=cv2.contourArea)

# find the extreme points
extLeft = tuple(c[c[:, :, 0].argmin()][0])
extRight = tuple(c[c[:, :, 0].argmax()][0])
extTop = tuple(c[c[:, :, 1].argmin()][0])
extBot = tuple(c[c[:, :, 1].argmax()][0])

# add contour on the image
img_cnt = cv2.drawContours(img.copy(), [c], -1, (0, 255, 255), 4)

# add extreme points
img_pnt = cv2.circle(img_cnt.copy(), extLeft, 8, (0, 0, 255), -1)
img_pnt = cv2.circle(img_pnt, extRight, 8, (0, 255, 0), -1)
img_pnt = cv2.circle(img_pnt, extTop, 8, (255, 0, 0), -1)
img_pnt = cv2.circle(img_pnt, extBot, 8, (255, 255, 0), -1)

    # crop
ADD_PIXELS = 0
new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
new_img.resize(224, 224, 3)

img = new_img.reshape((1, 224, 224, 3))
result = np.argmax(model.predict(img))

print(result)