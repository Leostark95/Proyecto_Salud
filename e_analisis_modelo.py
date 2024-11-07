import numpy as np
import joblib ### para cargar array

########Paquetes para NN #########
import tensorflow as tf
from sklearn import metrics ### para analizar modelo
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import a_funciones as fn
from sklearn.metrics import roc_curve

###### Datos #####
x_train, y_train, x_test, y_test, x_val, y_val = fn.imag_array()


#### Escalar ######################
x_train=x_train.astype('float32') ## para poder escalarlo
x_test=x_test.astype('float32') 
x_val = x_val.astype('float32')

x_train /=255 
x_test /=255
x_val /=255


##### cargar modelo  ######

modelo=tf.keras.models.load_model('salidas/best_model.keras')


####desempeño en evaluación para grupo 1 (tienen cancer) #######
prob=modelo.predict(x_val)
sns.histplot(prob, legend=False)
plt.title("probabilidades imágenes en validación")### conocer el comportamiento de las probabilidades para revisar threshold


threshold_mal=0.8

pred_val=(modelo.predict(x_val)>=threshold_mal).astype('int')
print(metrics.classification_report(y_val, pred_val))
cm = metrics.confusion_matrix(y_val, pred_val, labels=[0, 1])
disp = metrics.ConfusionMatrixDisplay(cm, display_labels=['Benigno', 'Maligno'])
disp.plot()



### desempeño en entrenamiento #####
prob=modelo.predict(x_train)
sns.histplot(prob, legend=False)
plt.title("probabilidades imágenes en entrenamiento")### conocer el comportamiento de las probabilidades para revisar threshold

pred_train=(prob>=threshold_mal).astype('int')
print(metrics.classification_report(y_train, pred_train))
cm = metrics.confusion_matrix(y_train, pred_train, labels=[0, 1])
disp = metrics.ConfusionMatrixDisplay(cm, display_labels=['Benigno', 'Maligno'])
disp.plot()


########### ##############################################################
####desempeño en evaluación para grupo 0 (No tienen cáncer) #######
########### ##############################################################

prob=modelo.predict(x_val)
sns.histplot(prob, legend=False)
plt.title("probabilidades imágenes en validación")### conocer el comportamiento de las probabilidades para revisar threshold

threshold_b=0.3 

pred_val=(modelo.predict(x_val)>=threshold_b).astype('int')
print(metrics.classification_report(y_val, pred_val))
cm = metrics.confusion_matrix(y_val, pred_val, labels=[0, 1])
disp = metrics.ConfusionMatrixDisplay(cm, display_labels=['Benigno', 'Maligno'])
disp.plot()


### desempeño en entrenamiento #####
prob=modelo.predict(x_train)
sns.histplot(prob, legend=False)
plt.title("probabilidades imágenes en entrenamiento")### conocer el comportamiento de las probabilidades para revisar threshold

pred_train = (prob >= threshold_b).astype('int')
print(metrics.classification_report(y_train, pred_train))
cm = metrics.confusion_matrix(y_train, pred_train, labels=[0, 1])
disp = metrics.ConfusionMatrixDisplay(cm, display_labels=['Benigno', 'Maligno'])
disp.plot()



####### clasificación final ################

prob=modelo.predict(x_val)

clas=['Prob alta' if prob >0.8 else 'Prob baja' if prob <0.3 else "Prob media" for prob in prob]

clases, count =np.unique(clas, return_counts=True)

count*100/np.sum(count)