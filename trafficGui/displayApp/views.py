from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from tensorflow.keras.models import load_model
from PIL import Image
import pandas as pd
import numpy as np
import cv2

model = load_model('./models/model_trained_b.h5')

#dictionary to label all traffic signs class.
classes = { 1:'Speed limit (20km/h)',
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

# Create your views here.
def index(request):
    context={'a':1}
    return render(request, 'index.html', context)

def actual(fileName):
    df  = pd.read_csv('./models/Test.csv')
    if '_' in fileName:
        fileName = ('Test/'+fileName).split('_')[0]+'.png'
    else:
        fileName = 'Test/'+fileName
    print('fileName in actual, ', fileName)

    actual_sign = df.loc[df['Path']==fileName]['ClassId'].values[0]
    print('actual_sign ', actual_sign )
    actual_sign = classes[int(actual_sign)+1]
    print('actual_sign' ,actual_sign)
    return actual_sign

def predictImage(request):
    fileObj = request.FILES['filePath']
    fs = FileSystemStorage()
    fileName = fs.save(fileObj.name,fileObj)
    filePathName = fs.url(fileName)
    print(filePathName)
    print(fileName)

    def grayscale(img):
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        return img
    def equalize(img):
        img =cv2.equalizeHist(img)
        return img
    def preprocessing(img):
        img = grayscale(img)
        img = equalize(img)
        img = img/255
        return img

    imgOriginal = Image.open('.'+filePathName)
    img = np.asarray(imgOriginal)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    cv2.imshow("Processed Image", img)
    img = img.reshape(1, 32, 32, 1)

    pred = model.predict_classes([img])[0]
    sign = classes[pred+1]
    actual_sign_name = actual(fileName)

    if actual_sign_name==sign:
        text_color='text-success'
    else:
        text_color='text-danger'


    context={'filePathName':filePathName,'imageFile':fileName,'predictedSign':sign,'actualSign':actual_sign_name,'text_color':text_color}
    return render(request, 'indexReload.html', context)

def predictVideo(request):
    #context={'a':1}
    model = load_model('./models/model_trained_b.h5')

#############################################
    frameWidth= 640         # CAMERA RESOLUTION
    frameHeight = 480
    brightness = 180
    threshold = 0.75         # PROBABLITY THRESHOLD
    font = cv2.FONT_HERSHEY_SIMPLEX
##############################################

# SETUP THE VIDEO CAMERA
    cap = cv2.VideoCapture(0)
    cap.set(3, frameWidth)
    cap.set(4, frameHeight)
    cap.set(10, brightness)



    def grayscale(img):
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        return img
    def equalize(img):
        img =cv2.equalizeHist(img)
        return img
    def preprocessing(img):
        img = grayscale(img)
        img = equalize(img)
        img = img/255
        return img
    def getCalssName(classNo):
        if   classNo == 0: return 'Speed Limit 20 km/h'
        elif classNo == 1: return 'Speed Limit 30 km/h'
        elif classNo == 2: return 'Speed Limit 50 km/h'
        elif classNo == 3: return 'Speed Limit 60 km/h'
        elif classNo == 4: return 'Speed Limit 70 km/h'
        elif classNo == 5: return 'Speed Limit 80 km/h'
        elif classNo == 6: return 'End of Speed Limit 80 km/h'
        elif classNo == 7: return 'Speed Limit 100 km/h'
        elif classNo == 8: return 'Speed Limit 120 km/h'
        elif classNo == 9: return 'No passing'
        elif classNo == 10: return 'No passing for vechiles over 3.5 metric tons'
        elif classNo == 11: return 'Right-of-way at the next intersection'
        elif classNo == 12: return 'Priority road'
        elif classNo == 13: return 'Yield'
        elif classNo == 14: return 'Stop'
        elif classNo == 15: return 'No vechiles'
        elif classNo == 16: return 'Vechiles over 3.5 metric tons prohibited'
        elif classNo == 17: return 'No entry'
        elif classNo == 18: return 'General caution'
        elif classNo == 19: return 'Dangerous curve to the left'
        elif classNo == 20: return 'Dangerous curve to the right'
        elif classNo == 21: return 'Double curve'
        elif classNo == 22: return 'Bumpy road'
        elif classNo == 23: return 'Slippery road'
        elif classNo == 24: return 'Road narrows on the right'
        elif classNo == 25: return 'Road work'
        elif classNo == 26: return 'Traffic signals'
        elif classNo == 27: return 'Pedestrians'
        elif classNo == 28: return 'Children crossing'
        elif classNo == 29: return 'Bicycles crossing'
        elif classNo == 30: return 'Beware of ice/snow'
        elif classNo == 31: return 'Wild animals crossing'
        elif classNo == 32: return 'End of all speed and passing limits'
        elif classNo == 33: return 'Turn right ahead'
        elif classNo == 34: return 'Turn left ahead'
        elif classNo == 35: return 'Ahead only'
        elif classNo == 36: return 'Go straight or right'
        elif classNo == 37: return 'Go straight or left'
        elif classNo == 38: return 'Keep right'
        elif classNo == 39: return 'Keep left'
        elif classNo == 40: return 'Roundabout mandatory'
        elif classNo == 41: return 'End of no passing'
        elif classNo == 42: return 'End of no passing by vechiles over 3.5 metric tons'

    while True:

        # READ IMAGE
        success, imgOrignal = cap.read()

        # PROCESS IMAGE

        img = np.asarray(imgOrignal)
        print('img 1',img.shape)
        img = cv2.resize(img, (32, 32))
        print('img 2',img.shape)
        img = preprocessing(img)
        print('img 3',img.shape)
        cv2.imshow("Processed Image", img)
        img = img.reshape(1, 32, 32, 1)
        print('img 4',img.shape)


        pred = model.predict_classes([img])[0]
        sign = classes[pred+1]
        print(sign)


        cv2.putText(imgOrignal, "CLASS: " , (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgOrignal, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        # PREDICT IMAGE
        predictions = model.predict(img)
        classIndex = model.predict_classes(img)
        probabilityValue =np.amax(predictions)
        if probabilityValue > threshold:
            #print(getCalssName(classIndex))
            cv2.putText(imgOrignal,str(classIndex)+" "+str(getCalssName(classIndex)), (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(imgOrignal, str(round(probabilityValue*100,2) )+"%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("Result", imgOrignal)

        if cv2.waitKey(1) and 0xFF == ord('q'):
            break
