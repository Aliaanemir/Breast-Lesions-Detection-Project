#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, render_template, request, jsonify, make_response
from flask_cors import CORS
import cv2 
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
import os
import time


# In[2]:


# Internal features dictionary 
# Bengin
Duct_Ectasia_in = {'name':('Duct Ectasia', 'N60.4'), 'MassShape':'IRREGULAR',
                   'MassMargin':'OBSCURED', 'Pathology':'Benign'}

Intraductal_Papilloma_in = {'name':('Intraductal Papilloma', 'N60.1'), 'MassShape':'LOBULATED',
                            'MassMargin':'CIRCUMSCRIBED', 'Pathology':'Benign'}

Mastitis_in = {'name':('Mastitis', 'N61.0'), 'MassShape':'IRREGULAR',
               'MassMargin':'ILL_DEFINED', 'Pathology':'Benign'}

Fibroadenomas_in = {'name':('Intraductal Papilloma', 'N60. 2'), 'MassShape':'OVAL',
                            'MassMargin':'CIRCUMSCRIBED', 'Pathology':'Benign'}

# Malignant
Invasive_Lobular_carcinoma_in = {'name':('Invasive Lobular Carcinoma', 'C50.9'), 'MassShape':'IRREGULAR', 
                                 'MassMargin':'SPICULATED','Pathology':'Malignant'}

Tubular_carcinoma_in = {'name':('Tubular Carcinoma', 'C50.9'), 'MassShape':'IRREGULAR',
                        'MassMargin':'SPICULATED','Pathology':'Malignant'}

Invasive_papillary_carcinoma_in = {'name':('Invasive Papillary Carcinoma', 'C73'), 'MassShape':'OVAL',
                        'MassMargin':'CIRCUMSCRIBED','Pathology':'Malignant'}


# In[3]:


# Internal features dictionary 
# Benign
Duct_Ectasia_ex = {'name':('Duct Ectasia', 'N60.4'), 

                   
                   'AgeGroup': 'y45-65',
                   
                   'PalbableMass':'Yes',
                   'NippleRetraction':'Yes',
                   'BreastItchiness':'No',
                   'BreastScaling_Dimpling':'No',
                   'BreastSoreness_Tenderness':'Yes',
                   'BreastSwelling_Enlargement':'No',
                   'BreastShrinkage':'No',

                   'NippleDischarge':'Yes',                   
                   'PregnancyHistory':'Yes',                    
                   'BreastfeedingHistory':'Yes',                   
                   'ConnectiveTissueDiseaseHistory':'Yes',                    
                   'Smoking':'Yes',  
                   'Fever': 'No'
                  }


Intraductal_Papilloma_ex = {'name':('Intraductal Papilloma', 'N60.1'),
                            
                            
                   'AgeGroup': 'y25-45',
                        
                   'PalbableMass':'Yes',
                   'NippleRetraction':'No',
                   'BreastItchiness':'No',
                   'BreastScaling_Dimpling':'No',
                   'BreastSoreness_Tenderness':'No',
                   'BreastSwelling_Enlargement':'Yes',
                   'BreastShrinkage':'No',       

                   'NippleDischarge':'Yes',                            
                   'PregnancyHistory':'No',                            
                   'BreastfeedingHistory':'No',                            
                   'ConnectiveTissueDiseaseHistory':'No',                    
                   'Smoking':'No',
                   'Fever':'No'                            
                  }

Mastitis_ex = {'name':('Mastitis', 'N61.0'),
                            
                            
                   'AgeGroup': 'y25-45',
                        
                   'PalbableMass':'No',
                   'NippleRetraction':'No',
                   'BreastItchiness':'No',
                   'BreastScaling_Dimpling':'No',
                   'BreastSoreness_Tenderness':'No',
                   'BreastSwelling_Enlargement':'Yes',
                   'BreastShrinkage':'No',       

                   'NippleDischarge':'No',                            
                   'PregnancyHistory':'No',                            
                   'BreastfeedingHistory':'Yes',                            
                   'ConnectiveTissueDiseaseHistory':'No',                    
                   'Smoking':'No',
                   'Fever': 'Yes'                           
                  }

Fibroadenomas_ex = {'name':('Duct Ectasia', 'N60.2'), 

                   
                   'AgeGroup': 'y45-65',
                   
                   'PalbableMass':'Yes',
                   'NippleRetraction':'No',
                   'BreastItchiness':'No',
                   'BreastScaling_Dimpling':'No',
                   'BreastSoreness_Tenderness':'No',
                   'BreastSwelling_Enlargement':'Yes',
                   'BreastShrinkage':'No',

                   'NippleDischarge':'No',                   
                   'PregnancyHistory':'No',                    
                   'BreastfeedingHistory':'No',                   
                   'ConnectiveTissueDiseaseHistory':'No',                    
                   'Smoking':'Yes',
                   'Fever': 'No'
                  }


# Malignant
Invasive_Lobular_carcinoma_ex = {'name':('Invasive Lobular Carcinoma', 'C50.9'),
                   'AgeGroup': 'y45-65',
                   
                   'PalbableMass':'No',
                   'NippleRetraction':'Yes',
                   'BreastItchiness':'Yes',
                   'BreastScaling_Dimpling':'Yes',
                   'BreastSoreness_Tenderness':'No',
                   'BreastSwelling_Enlargement':'Yes',
                   'BreastShrinkage':'Yes',

                   'NippleDischarge':'Yes',
                   'PregnancyHistory':'No',
                   'BreastfeedingHistory':'No',
                   'ConnectiveTissueDiseaseHistory':'No',
                   'Smoking':'No',
                   'Fever': 'No'
                                 
                  }

Tubular_Carcinoma_ex = {'name':('Tubular Carcinoma', 'C50.9'),
                   'AgeGroup': 'y45-65',
                        
                   'PalbableMass':'No',
                   'NippleRetraction':'Yes',
                   'BreastItchiness':'No',
                   'BreastScaling_Dimpling':'Yes',
                   'BreastSoreness_Tenderness':'No',
                   'BreastSwelling_Enlargement':'No',
                   'BreastShrinkage':'No',

                   'NippleDischarge':'Yes',
                   'PregnancyHistory':'No',
                   'BreastfeedingHistory':'No',
                   'ConnectiveTissueDiseaseHistory':'No',
                   'Smoking':'No',
                   'Fever': 'No'
                        
                  }

Invasive_Papillary_Carcinoma_ex = {'name':('Invasive Papillary Carcinoma', 'C73'),
                   'AgeGroup': 'y45-65',
                        
                   'PalbableMass':'Yes',
                   'NippleRetraction':'No',
                   'BreastItchiness':'Yes',
                   'BreastScaling_Dimpling':'No',
                   'BreastSoreness_Tenderness':'No',
                   'BreastSwelling_Enlargement':'No',
                   'BreastShrinkage':'No',

                   'NippleDischarge':'Yes',
                   'PregnancyHistory':'No',
                   'BreastfeedingHistory':'No',
                   'ConnectiveTissueDiseaseHistory':'No',
                   'Smoking':'No',
                   'Fever': 'No'
                  }


# In[4]:


lesions_in = pd.DataFrame([Duct_Ectasia_in, Intraductal_Papilloma_in, Mastitis_in,
                           Fibroadenomas_in, Invasive_Lobular_carcinoma_in, Tubular_carcinoma_in])

lesions_ex = pd.DataFrame([Duct_Ectasia_ex, Intraductal_Papilloma_ex,Mastitis_ex, Fibroadenomas_ex,
                           Invasive_Lobular_carcinoma_ex,Tubular_Carcinoma_ex, Invasive_Papillary_Carcinoma_ex])


# lesions data 
lesions_data = pd.merge(left= lesions_in, right= lesions_ex)
index = pd.MultiIndex.from_tuples(lesions_data['name'], names=["name", "code"])
lesions_data.set_index(index, inplace=True)
lesions_data.drop('name', axis=1, inplace=True)


# In[5]:


def compute_score(patient_data, lesions_features, breast_site):
    
    scoring_matrix = {}
    
    ## twice the points for pathology, half a point for uncertanity and errors
    max_score= (len(lesions_features.columns)-1)*1 + 2 + 0.5

    for lesion in lesions_features.index:
        lesion_data=lesions_features.loc[lesion]
        score = 0
        #comparing and scoring
        result = (patient_data == lesion_data)
        for feature in result.keys():
            if result[feature] == True:
                if feature == 'pathology':
                    score += 2
                else:
                    score += 1
        probabilty = (score/max_score)
        scoring_matrix[lesion] = probabilty

    #sort vs sorted
    sorted_scores = sorted(scoring_matrix.items(), key=lambda x: x[1], reverse=True)
    #sort and print the top 3 lesions 
    '''pathology = ("Most probably {} with {:.2f}% accuracy for clinico-pathological correlation").format(sorted_scores[0][0], sorted_scores[0][1])
    DD = "Differential Diagnosis:  "
    print("Differential Diagnosis:")'''

    result = {}
    result['MostProbably']= {'Diagnosis': sorted_scores[0][0][0], 'Accuracy': str(round(sorted_scores[0][1],2)), 'ChildCode': sorted_scores[0][0][1], 'BreastSite': breast_site} 
    result['DifferentialDiagnosis'] = []
    for i in range(1,3):
     
        result['DifferentialDiagnosis'].append({'DifferentialDiagnosis_'+str(i): sorted_scores[i][0][0], 'Accuracy': str(round(sorted_scores[i][1],2)), 'ChildCode': sorted_scores[i][0][1]})
    
    return result


# In[6]:


def clahe(img, clip=2.0, tile=(8, 8)):

    img = cv2.normalize(
        img,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_32F,
    )
    img_uint8 = img.astype("uint8")

    clahe_create = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    clahe_img = clahe_create.apply(img_uint8)

    return clahe_img

def resnet_prep(image):
    resized = cv2.resize(image, (244, 244))
    equalized = clahe(resized)
    image = np.dstack((equalized, resized, equalized))
    return image
    
    
def minMaxNormalise(img):
    norm_img = img/(2**16)
    return norm_img


def mass_margin_prep(img):
    normalized = minMaxNormalise(img)
    resized = cv2.resize(normalized, (299, 299))
    equalized = clahe(resized)/255
    return np.array(equalized)


# In[7]:


def get_input():
    
    patient_data = {}
    
    # EXTRACTING INTERNAL FEATURES 
    # Tumor classification
    labels_pathology = ['Normal', 'Benign', 'Malignant']
    
    #save input
    
    imagefile = request.files['TumorImage']
    image_path = '/'.join([target, imagefile.filename])
    imagefile.save(image_path)
    
    #preprocess input
    sample= cv2.imread(image_path, 0)
    image = image_prep(sample)
    
    #prediction
    out = model_pathology.predict(image)
    max_idx = np.argmax(out,axis=1)
    label = labels_pathology[max_idx[0]]
    #print("Tumor is: " + label)
    
    # Mass Shape
    labels_shape = ['ARCHITECTURAL_DISTORTION', 'IRREGULAR',
       'IRREGULAR-ARCHITECTURAL_DISTORTION', 'LOBULATED', 'OVAL', 'ROUND']
    
    test_datagen = ImageDataGenerator(rescale=1./2**16)
    
    image = resnet_prep(sample)
    x = image.reshape(1,244,244,3)
    generator = test_datagen.flow(x, batch_size=1)
    
    
    #prediction
    out = model_shape.predict(generator)
    max_idx = np.argmax(out,axis=1)
    shape = labels_shape[max_idx[0]]
    
    #print("Mass Shape is: " + shape)
    
    
    #MASS MARGIN#
    labels_margin = ['CIRCUMSCRIBED', 'ILL_DEFINED', 'MICROLOBULATED', 'OBSCURED', 'SPICULATED']
    
    image = mass_margin_prep(sample)
    x = image.reshape(1,299,299,1)
    
    #prediction
    out = model_margin.predict(x)
    max_idx = np.argmax(out,axis=1)
    margin = labels_margin[max_idx[0]]
    #print("Tumor is: " + label)  
    
     
    #INTERNAL FEATURES
    patient_data['MassShape'] = shape
    patient_data['MassMargin'] = margin
    patient_data['Pathology'] = label
    
    # PARSING EXTERNAL FEATURES 
    
    patient_data['AgeGroup'] = request.form['AgeGroup']
    patient_data['PalbableMass'] = request.form['PalbableMass']
    patient_data['NippleRetraction'] = request.form['NippleRetraction']
    patient_data['BreastItchiness'] = request.form['BreastItchiness']
    patient_data['BreastScaling_Dimpling'] = request.form['BreastScaling_Dimpling']
    patient_data['BreastSoreness_Tenderness'] = request.form['BreastSoreness_Tenderness']
    patient_data['BreastSwelling_Enlargement'] = request.form['BreastSwelling_Enlargement']
    patient_data['BreastShrinkage'] = request.form['BreastShrinkage']
    patient_data['NippleDischarge'] = request.form['NippleDischarge']
    patient_data['PregnancyHistory'] = request.form['PregnancyHistory']
    patient_data['BreastfeedingHistory'] = request.form['BreastfeedingHistory']
    patient_data['ConnectiveTissueDiseaseHistory'] = request.form['ConnectiveTissueDiseaseHistory']
    patient_data['Smoking'] = request.form['Smoking']
    patient_data['Fever'] = request.form['Fever']
    
    # METADATA
    breast_site = request.form['BreastSite']
    
    return pd.Series(patient_data), breast_site


# In[8]:


def image_prep(image):
    resized = cv2.resize(image, (100, 100))
    reshaped = resized.reshape(1,100,100,1)
    return reshaped


# In[9]:


# app

APP_ROOT = os.path.dirname(os.path.abspath('app_model_response.py'))

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
CORS(app)

target = os.path.join(APP_ROOT, 'static')
if not os.path.isdir(target):
    os.mkdir(target)

global model_pathology
global model_shape
global model_margin


model_pathology = load_model('Tumor_Classification.h5', compile=False)
model_shape = load_model('Mass_Shape_Detection.h5', compile=False)
model_margin = load_model('Mass_Margin_Detection_balanced_clean.h5', compile=False)

labels_pathology = ['Normal', 'Benign', 'Malignant']
labels_shape = ['ARCHITECTURAL_DISTORTION', 'IRREGULAR',
       'IRREGULAR-ARCHITECTURAL_DISTORTION', 'LOBULATED', 'OVAL', 'ROUND']
labels_margin = ['CIRCUMSCRIBED', 'ILL_DEFINED', 'MICROLOBULATED', 'OBSCURED', 'SPICULATED']

@app.route('/diagnosis', methods=['POST'])
def model_reponse():
    start = time.time()
    patient_data, breast_site = get_input()
    result = make_response(jsonify(compute_score(patient_data, lesions_data, breast_site), {'Time':round(time.time()-start+0.3,1)}))
    
    return result

@app.route('/predict-tumor', methods=['POST'])
def predict_tumor():
    #save input
    imagefile = request.files['TumorImage']
    image_path = './images' + imagefile.filename
    imagefile.save(image_path)
    
    #preprocess input
    sample= cv2.imread(image_path, 0)
    image = image_prep(sample)
    
    #prediction
    out = model_pathology.predict(image)
    max_idx = np.argmax(out,axis=1)
    label_pathology = labels_pathology[max_idx[0]]

    result = make_response(jsonify({'prediction':label_pathology}))
   
    return result
 
    
@app.route('/predict-shape', methods=['POST'])
def predict_shape():
    #save input
    imagefile = request.files['TumorImage']
    image_path = './images' + imagefile.filename
    imagefile.save(image_path)
    
    #preprocess input
    test_datagen = ImageDataGenerator(rescale=1./2**16)
    
    sample= cv2.imread(image_path, 0)
    image = resnet_prep(sample)
    x = image.reshape(1,244,244,3)
    generator = test_datagen.flow(x, batch_size=1)
    
    
    #prediction
    out = model_shape.predict(generator)
    max_idx = np.argmax(out,axis=1)
    shape = labels_shape[max_idx[0]]

    result = make_response(jsonify({'prediction':shape}))
   


    return result

@app.route('/predict-margin', methods=['POST'])
def predict_margin():
    #save input
    imagefile = request.files['TumorImage']
    image_path = './images' + imagefile.filename
    imagefile.save(image_path)
    
    #preprocess input
    sample= cv2.imread(image_path, 0)
    image = mass_margin_prep(sample)
    x = image.reshape(1,299,299,1)
    
    #prediction
    out = model_margin.predict(x)
    max_idx = np.argmax(out,axis=1)
    margin = labels_margin[max_idx[0]]
    #print("Tumor is: " + label)

    result = make_response(jsonify({'prediction':margin}))
    return result
    
# In[ ]:


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=9000)


# In[ ]:




