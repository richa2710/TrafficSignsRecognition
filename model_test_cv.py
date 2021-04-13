import numpy as np 
import pandas as pd 
from PIL import Image
import cv2
from keras.models import load_model



def grayscale(img):
   img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
   return img
def equalize(img):
   img =cv2.equalizeHist(img)
   return img
def preprocessing(img):
   img = grayscale(img)     # CONVERT TO GRAYSCALE
   img = equalize(img)      # STANDARDIZE THE LIGHTING IN AN IMAGE
   img = img/255            # TO NORMALIZE VALUES BETWEEN 0 AND 1 INSTEAD OF 0 TO 255
   return img

#testing accuracy on test dataset

model = load_model('model_trained_b.h5')

y_test = pd.read_csv('Test.csv')

labels = y_test["ClassId"].values
imgs = y_test["Path"].values

data=[]
pred = []
for img in imgs:
    image = Image.open(img)
    image = np.asarray(image)
    image = cv2.resize(image, (32, 32))
    image = preprocessing(image)
    image = image.reshape(1, 32, 32, 1)
    #data.append(np.array(image))
    image = np.array(image)
    #print("Image array,", image.shape)
    #print(model.predict_classes(image)[0])
    pred.append(model.predict_classes(image)[0])


print('Total number of images in test dataset =',len(pred))

from sklearn.metrics import accuracy_score
print('Test accuracy =',accuracy_score(labels, pred))

#Number of correctly and incorrectly identified traffic signs
index=0
misclassifiedIndex=[]
correctlyClassifiedIndex=[]
for predict,actual in zip(pred,labels):
    if predict!=actual:
        misclassifiedIndex.append(index)
    elif predict==actual:
        correctlyClassifiedIndex.append(index)
    index+=1
print('Number of incorrectly identified signs = ',len(misclassifiedIndex))
print('Number of correctly identified signs = ',len(correctlyClassifiedIndex))

yfile = y_test['Path']
print('List of incorrectly identified signs:\n',yfile[misclassifiedIndex])


