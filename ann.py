
#Importar librerias
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Codificar datos categóricos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [1])],   
    remainder='passthrough'                        
)
X = onehotencoder.fit_transform(X)
X = X[:, 1:]

#Dividir el data set en conjunto de entrenamiento y en conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Escalando Variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Construir la RNA
#Importar keras y librerias  adicionales

import keras
from keras.models import Sequential
from keras.layers import Dense

#Inicializar la red neuronal

classifier = Sequential()
#Añadir capas de entrada  y primera capa oculta
classifier.add(Dense(units = 6, kernel_initializer="uniform", activation ="relu", input_dim = 11))
#Añadir una segunda capa oculta
classifier.add(Dense(units = 6, kernel_initializer="uniform", activation ="relu"))
#Añadir la capa de salida
classifier.add(Dense(units = 1, kernel_initializer="uniform", activation ="sigmoid"))
#Compilar la RNA
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
#Ajustamos la red neuronal al conjunto de entrenamiento
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

#Evaluar el modelo y calcular las predicciones finales
#Prediccion de los resultados del conjunto de testing
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)
#Elaborar una matriz de confusion
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


