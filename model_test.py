import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical



data = []
labels = []
classes = 43
cur_path = os.getcwd()
print(cur_path)


#dictionary to label all traffic signs class.

sign_names={1:'Speed limit (20km/h)',
            2:'Speed limit (30km/h)',      
            3:'Speed limit (50km/h)',       
            4:'Speed limit (60km/h)',      
            5:'Speed limit (70km/h)',    
            6:'Speed limit (80km/h)',      
            7:'End of speed limit (80km/h)',     
            8:'Speed limit (100km/h)',    
            9:'Speed limit (120km/h)',     
           10:'No passing',   
           11:'No passing veh over 3.5 tons',     
           12:'Right-of-way at intersection',     
           13:'Priority road',    
           14:'Yield',     
           15:'Stop',       
           16:'No vehicles',       
           17:'Veh > 3.5 tons prohibited',       
           18:'No entry',       
           19:'General caution',     
           20:'Dangerous curve left',      
           21:'Dangerous curve right',   
           22:'Double curve',      
           23:'Bumpy road',     
           24:'Slippery road',       
           25:'Road narrows on the right',  
           26:'Road work',    
           27:'Traffic signals',      
           28:'Pedestrians',     
           29:'Children crossing',     
           30:'Bicycles crossing',       
           31:'Beware of ice/snow',
           32:'Wild animals crossing',      
           33:'End speed + passing limits',      
           34:'Turn right ahead',     
           35:'Turn left ahead',       
           36:'Ahead only',      
           37:'Go straight or right',      
           38:'Go straight or left',      
           39:'Keep right',     
           40:'Keep left',      
           41:'Roundabout mandatory',     
           42:'End of no passing',      
           43:'End no passing veh > 3.5 tons' }


#Retrieving the images and their labels 
plt.figure(figsize=(20,5))

im_read = 0
im_error = 0
index = 1
for i in range(classes):
    path = os.path.join(cur_path,'Train',str(i))
    images = os.listdir(path)
    fignum=0
    for a in images:
        #print(a)
        try:
            fignum+=1
            image = Image.open(path + '/'+ a)
            if i==1 or i==12 or i==21 or i==35 or i==41:
                if fignum==1:
                    plt.subplot(1,5,index)
                    plt.imshow(image)
                    sign_name = sign_names[i+1]
                    plt.title(sign_name,fontsize=20)
                    plt.axis('off')
                    index+=1
            image = image.resize((30,30))
            image = np.array(image)
            data.append(image)
            labels.append(i)
            im_read+=1
            
        except:
            print("Error loading image")
            im_error+=1
plt.suptitle('Example Images of Traffic Signs',fontsize=25)
print('\nImages read = ', im_read)
print('Error reading images = ', im_error)


#Converting lists into numpy arrays
data = np.array(data)
labels = np.array(labels)
print(data)
print('\n',labels)


print(data.shape, labels.shape)

print(type(data))


#Splitting training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#Converting the labels into one hot encoding
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)


print(y_train.shape)

#testing accuracy on test dataset
from keras.models import load_model

model = load_model('my_model.h5')

y_test = pd.read_csv('Test.csv')

labels = y_test["ClassId"].values
imgs = y_test["Path"].values

data=[]

for img in imgs:
    image = Image.open(img)
    image = image.resize((30,30))
    data.append(np.array(image))

X_test=np.array(data)

pred = model.predict_classes(X_test)


print('Total number of images in test dataset =',len(X_test))


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


