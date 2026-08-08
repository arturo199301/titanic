import numpy as np
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier


st.write('''sobrevivientes del Titanic ''')
st.image("images.jpg", caption="El Titanic navegaba desde Southampton, Inglaterra, hasta Nueva York en Estados Unidos.")

st.header('Datos de evaluación')

def user_input_features():
  # Entrada
  Pclass = st.number_input('Clase:', min_value=1, max_value=3, value = 1, step = 1)
  Sex = st.number_input('Género:', min_value=0, max_value=1, value = 0, step = 1)
  Age = st.number_input('Edad:', min_value=0, max_value=100, value = 0, step = 1)
  SibSp = st.number_input('Hermanos(as)/Esposo(a):',min_value=0, max_value=10, value = 0, step = 1)
  Parch = st.number_input('Padres/Hijos:', min_value=0, max_value=10, value = 0, step = 1)
  Fare = st.number_input('Tarifa:')
  Embarked = st.number_input('Lugar de Embarque:', min_value=0, max_value=2, value = 0, step = 1)

  user_input_data = {'Pclass': Pclass,
                     'Sex': Sex,
                     'Age': Age,
                     'SibSp': SibSp,
                     'Parch': Parch,
                     'Fare': Fare,
                     'Embarked': Embarked}

  features = pd.DataFrame(user_input_data, index=[0])

  return features

df = user_input_features()

titanic =  pd.read_csv('Titanic2.csv', encoding='latin-1')
X = titanic.drop(columns='Survived')
Y = titanic['Survived']

classifier = DecisionTreeClassifier(max_depth=8, criterion='entropy', min_samples_leaf=10, max_features=7, random_state=0)
classifier.fit(X, Y)

prediction = classifier.predict(df)

st.subheader('Predicción')
if prediction == 0:
  st.write('No sobrevive')
elif prediction == 1:
  st.write('Sobrevive')
else:
  st.write('Sin predicción')
