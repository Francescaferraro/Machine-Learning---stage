#mobilenet + linearSVM (T3)

import os.path
from deepface import DeepFace
from os import listdir
from os.path import isfile
import tensorflow as tf
import numpy as np
import random
import cv2
import re
import os
# TensorFlow and TF-Hub modules.
from absl import logging
import tensorflow as tf
import tensorflow_hub as hub
logging.set_verbosity(logging.ERROR)
from tensorflow import keras
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import imblearn
from imblearn.under_sampling import RandomUnderSampler
#modifica per evitare ModuleNotFoundError
file = "/home/studenti/ferraro/.local/lib/python3.8/site-packages/keras_vggface/models.py"
text = open(file).read()
open(file, "w+").write(text.replace('keras.engine.topology', 'tensorflow.keras.utils'))

from sklearn.metrics import accuracy_score
import statistics
from statistics import mode

base_dir = '/var/datasets/VHR1/'
IMG_SIZE = 224
NUM_FEATURES = 1280
num_frame = 6325 #6325 numero di frame presenti nei video
groups = []
num_video=0
groups_video = []
lista_video = [] #contiene le features per ogni frame del video
num_subjects = 56 #numero totale di soggetti nel dataset
labels_tot=[] #insieme delle labels target del dataset, una per ogni video

#organizzazione file sul server: separati in base alla task e al soggetto
type_dir = base_dir+'UBFC_Phys_T3'

with tf.device('/cpu:0'): #forza l'uso della cpu in keras
    #face detector on video frames
    def load_video(path, numero, video, resize=(IMG_SIZE, IMG_SIZE)) : 
        print(path)
        labels = [] #contiene i frame classificati come 0 o 1
        cap = cv2.VideoCapture(path)
        frames = []
        count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if count<=num_frame :
                    print(count)
                    detectors = ["opencv", "ssd", "mtcnn", "dlib", "retinaface"]
                    #mtcnn
                    try :
                        frame_ritagliato = DeepFace.detectFace(frame, detector_backend = detectors[2])
                        frames.append(frame_ritagliato) #aggiunge solo i frame di cui rileva il volto
                        groups.append(numero)
                        count=count+35

                        #aggiunge un 1 o 0 all'array delle labels relativo al video in questione
                        print(video)
                        video_dir = type_dir+'/s'+str(video)
                        informazioni = video_dir+'/info_s'+str(video)+'.txt'
                        filename = video_dir+'/vid_s'+str(video)+'_T3.avi'
                        
                        i = 0
                        with open(informazioni, 'r') as f:
                          for line in f:
                              i=i+1
                              
                              j=0
                              if (i == 3) and (line == 'test\n') : #se il video è acquisito in condizioni di "test", tutti i frame sono 'stress'
                                labels.append(1)  
                                break
                              if ((i == 3) and (line == 'ctrl\n')) : #se il video è acquisito in condizioni "ctrl", tutti i frame sono 'non stress'
                                labels.append(0)
                                break
                        #print(labels)
                        #print(groups)

                    except :
                        print("ex")
                        count=count+1
                    
                else :
                    break
                
        finally:
            cap.release()

        if (len(labels)>0) : #se c'è almeno un frame validato
          labels_tot.append(labels) #aggiunge a labels_tot solo i gruppi di labels che contengono effettivamente frame

        return np.array(frames)

  
    #estrazione di features
    #building the 3D-CNN model
    def build_feature_extractor() :
        
        feature_extractor_model = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
        
        feature_extractor_layer = hub.KerasLayer(
          feature_extractor_model,
          input_shape=(224, 224, 3),
          trainable = False
          )

        inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3)) #instantiate a Keras tensor
        outputs = feature_extractor_layer(inputs)

        keras.Model(inputs, outputs, name="feature_extractor").summary()
        return keras.Model(inputs, outputs, name="feature_extractor")
        

    feature_extractor = build_feature_extractor()



    def prepare_video(path, numero, video):
        # Gather all video frames
        frames = load_video(path, numero, video)
        frame_features = np.zeros(shape=(1, len(frames), NUM_FEATURES), dtype="float32")
        
        # Initialize placeholders to store the features of the current video.
        temp_frame_features = np.zeros(shape=(1, len(frames), NUM_FEATURES), dtype="float32")
        #add a batch dimension
        frames = frames[None, :]
        
        print(temp_frame_features.shape)
        # Extract features from the frames of the current video.
        for i, batch in enumerate(frames):
            video_length = batch.shape[0]
            for j in range(video_length):
                temp_frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
        
        frame_features = temp_frame_features.squeeze() #change the shape of a three-dimensional array to a two-dimensional array

        return frame_features

    k = 1
    while k<=num_subjects :
        video_dir = type_dir+'/s'+str(k)
        info = video_dir+'/info_s'
        filename = video_dir+'/vid_s'+str(k)+'_T3.avi'
        frame_features = prepare_video(filename, num_video, k)
        lista_video.append(frame_features)
        if len(frame_features)>0 : #se c'è almeno un frame validato
            groups_video.append(num_video)
            num_video=num_video+1 #conto il numero totale di video
        else :
            lista_video.remove(frame_features) #se non ci sono frame validati, escludo il video
        print(f"Frame features in set: {frame_features.shape}")
        k=k+1

    labels_def=[]
    for x in labels_tot :
      print(x)
      for k in x :
        labels_def.append(k)

    lista=[]
    for k in lista_video :
      for j in k :
        lista.append(j)
     

    video_numero = {} #associa ad ogni frame il suo valore in groups, 
    #servirà per suddividere corettamente i frame nei vari video dopo la classificazione svm
    v = 0
    for k in groups :
      video_numero.update([(v, k)])
      v=v+1
    
    #StratifiedGroupKFold

    n_splits=7
    group_kfold = StratifiedGroupKFold(n_splits=n_splits)

    #svm Classifier
    clf = svm.SVC(kernel='linear') 
    accuracy=[]
    accuracy_video=[]

    lista = np.array(lista)
    labels_def = np.array(labels_def)

    # define undersample strategy
    undersample = RandomUnderSampler(sampling_strategy='majority')

    for train_index, test_index in group_kfold.split(X=lista, y=labels_def, groups=groups):
    #per considerare i singoli frame prendo lista che contiene tutte le features non suddivise, 
    #labels_def che contiene i gruppi di 0 o 1 non suddivisi e 
    #groups che contiene, per ogni frame, il numero identificativo del video a cui appartiene 
      X_train, X_test = lista[train_index], lista[test_index]
      y_train, y_test = labels_def[train_index], labels_def[test_index]
      
      #normalizzazione
      scaler = MinMaxScaler()
      scaler.fit(X_train)
      X_train = scaler.transform(X_train)
      X_test = scaler.transform(X_test)

      # fit and apply the transform
      X_train, y_train = undersample.fit_resample(X_train, y_train)
      X_test, y_test = undersample.fit_resample(X_test, y_test)

      #Train the model
      clf.fit(X_train, y_train)

      #Predict the response for test dataset
      y_pred = clf.predict(X_test)
      
    
      accuracy.append(accuracy_score(y_test, y_pred))
      print("Report frames\n", classification_report(y_test, y_pred, target_names=["non stress", "stress"]))
      print("Conf matrix frames\n", confusion_matrix(y_test, y_pred))

      #divido l'array di frame in subarray della dimensione dei video, così da averne uno per ogni video
      #poi calcolo la moda per ogni subarray, sia per test che per pred, e ne calcolo l'accuracy relativa

      s=[]
      k=0
      array_finale = []
      for j in test_index :
        k=k+1
        for i in video_numero.keys() :
          if i==j and video_numero.get(i)!=video_numero.get(i+1) :
            s.append(k) #s contiene l'indice in cui cambia video nel Test 
      x=0
      i=0
      video_test = []
      while i<len(s) :
        if i==0 :
          video_test = np.split(y_test, [s[i]])
          y_test = y_test[s[i]:]
        else :
          x = s[i]-s[i-1]
          video_test = np.split(y_test, [x])
          y_test = y_test[x:]
        array_finale.append(video_test[0])
        if i == len(s)-2 :
          array_finale.append(video_test[1])
          break
        i=i+1
      

      y_test_video = []
      y_pred_video = []
      j=0
      while j<len(array_finale) :
        if (len(array_finale[j])!=0) :
          moda_video_test = mode(array_finale[j])
          y_test_video.append(moda_video_test)
        j=j+1

      x=0
      i=0
      video_pred = []
      finale_pred = []
      while i<len(s) :
        if i==0 :
          video_pred = np.split(y_pred, [s[i]])
          y_pred = y_pred[s[i]:]
        else :
          x = s[i]-s[i-1]
          video_pred = np.split(y_pred, [x])
          y_pred = y_pred[x:]
        finale_pred.append(video_pred[0])
        if i == len(s)-2 :
          finale_pred.append(video_pred[1])
          break
        i=i+1
      
      
      i=0
      while i<len(finale_pred) :
        if (len(finale_pred[i])!=0) :
          moda_video_pred = mode(finale_pred[i])
          y_pred_video.append(moda_video_pred)
        i=i+1
      

      accuracy_video.append(accuracy_score(y_test_video, y_pred_video))
      print("Report video\n", classification_report(y_test_video, y_pred_video, target_names=["non stress", "stress"]))
      print("Conf matrix video\n", confusion_matrix(y_test_video, y_pred_video))

    #Predict the response for test dataset, for each frame
    print("Mean accuracy for frames: ", statistics.mean(accuracy))
    print("------------------------------------------------------------")
    #Predict the response for test dataset, for each video
    print("Mean accuracy for video: ", statistics.mean(accuracy_video))