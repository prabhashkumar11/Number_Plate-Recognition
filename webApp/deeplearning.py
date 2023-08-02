import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import pytesseract as pt

from tensorflow.keras.preprocessing.image import load_img, img_to_array
# Load Model
model=tf.keras.models.load_model('./static/models/object_detection.h5')



def object_detection(path,filename):
    image=load_img(path) #PIL object
    image=np.array(image,dtype=np.uint8)
    image1=load_img(path,target_size=(224,224))
    image_arr_224=img_to_array(image1)/225.0 #convert into array and get the normaliszed output
    h,w,d=image.shape
    test_arr=image_arr_224.reshape(1,224,224,3)   
    #make predictions
    coords=model.predict(test_arr)
    
    denorm=np.array([w,w,h,h])
    coords=coords*denorm
    
    coords=coords.astype(np.int32)
    
    xmin,xmax,ymin,ymax=coords[0]
    pt1=(xmin,ymin)
    pt2=(xmax,ymax)
    pt1,pt2
    cv2.rectangle(image,pt1,pt2,(0,255,0),3)
    # convert into bgr
    image_bgr=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    cv2.imwrite('./static/predict/{}'.format(filename),image_bgr)
    
    return coords



def OCR(path,filename):
    img=np.array(load_img(path))
    cods=object_detection(path,filename)
    xmin,xmax,ymin,ymax=cods[0]
    roi=img[ymin:ymax,xmin:xmax]
    roi_bgr=cv2.cvtColor(roi,cv2.COLOR_RGB2BGR)
    cv2.imwrite('./static/roi/{}'.format(filename),roi_bgr)
    # extract text from image
    text=pt.image_to_string(roi)
    print(text)
    return text
    