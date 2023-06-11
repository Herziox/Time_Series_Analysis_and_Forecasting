import streamlit as st

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os 
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

st.header('Predicción del precio de las acciones usando LTSM')
st.subheader('Conceptos')

st.text('''Serie de tiempo: valores capturados de un evento en determinado intervalor tiempo.
Por lo general son valores sucesivos en espacios de tiempo iguales. Estos intevalos 
de tiempo puede ser entre Años, Meses, Dias, Horas,Minutos, segundos o fracciones de segundo''')

st.text('''Análisis de series de tiempo: Consiste en la aplicación de métodos estadísticos para descubrir nuevas característticas
de las series de tiempo tales cómo patrones y descubrir nuevas perspectivas entorno a pronosticos más precisos''')

st.text('''Pronostico (Forecasting): Es la aplicación de tecnicas probabilisticas y estadísticas en base a valores pasados o actuales
y sus carácterísticas para predecir un valor futuro''')

st.subheader('Componentes')

st.text('''Tendencia: La tendencia muestra la dirección general de una serie de tiempo a lo largo de un periodo de tiempo''')

st.text('''Estacionalidad: Muestra la tendencia que se repite cada cierto tiempo''')

st.text('''Componente ciclico: Es un intervalo sin repetición donde se produce un movimiento o patron en la serie de tiempo ''')

st.text('''Irregularidad: Eventos inesperados en un corto periodo de tiempo''')


st.subheader('Tipos de Series de tiempo')
st.text('''Existen dos tipos de datos en las series de tiempo:''')
st.markdown('##Estacionario')
st.text('''Es el conjunto de datos que debe seguir las siguientes reglas sin tomar en cuenta la tendencia, estacionalidad, ciclicidad e irregularidad''')
st.caption('El valor medio (Promedio): debe ser constante')
st.caption('La varianza: debe ser constante con respecto al periodo de tiempo')
st.caption('La covarianza medira la relación entre dos variables')

st.markdown('##No Estacionario')
st.text('''Ocurre cuando  la varianza y la covarianza cambian con respecto al tiempo''')

 
st.header('Ejecucion del Modelo de LTSM')
 
st.text('Muestra de datos de cierre de las acciones de la empresa AAL')


plt.style.use('seaborn-white')
for param in ['figure.facecolor', 'axes.facecolor', 'savefig.facecolor']:
    plt.rcParams[param] = '#212946'  # bluish dark grey
for param in ['text.color', 'axes.labelcolor', 'xtick.color', 'ytick.color']:
    plt.rcParams[param] = '0.9'  # very light grey
plt.rcParams.update({"axes.grid": True,"grid.color": "#2A3459"})


#Carga de datos
df_indexProcessed = pd.read_csv(os.path.join('data.csv'))

#Seleccion de datos
df_alfa = df_indexProcessed[df_indexProcessed['symbol']=='AAL']

#Transformacion de datos a tipo Date
df_alfa['date'] = pd.to_datetime(df_alfa['date'])

#Eliminar columnas innecesarias
df_alfa.drop(columns=['symbol'],inplace=True)

#Utilizar solo la caracteristica que se busca predecir
df_data = df_alfa['close']

#Division de datos en entrenamiento y pruebas
train_size = int(len(df_alfa['close'])*0.65)
test_size = int(len(df_alfa['close'])*0.90)

scaler = MinMaxScaler(feature_range=(0,1))
df_data = scaler.fit_transform(np.array(df_data).reshape(-1,1))

train_data,test_data,validation_data = df_data[0:train_size,:],df_data[train_size:test_size,:1],df_data[test_size:len(df_alfa['close']),:1]




#transformacion de datos para su prediccion ya que LSTM utiliza una entrada de datos en 3D
def create_ds(dataset,time_steps):   
    data_x,data_y = [],[]
    for i in range(len(dataset)-time_steps-1):
        a = dataset[i:(i+time_steps),0]
        data_x.append(a)
        b = dataset[i+time_steps,0]
        data_y.append(b)
    return np.array(data_x),np.array(data_y)


#Graficar datos
st.line_chart(df_data)


st.subheader('Simulacion')
number = st.slider("Numero de dias a evaluar",
                           min_value=1,
                           max_value=200,)
st.metric("Dias a evaluar", number) #widgets help us declutter the application 

time_step = 50


x_input=test_data[len(test_data)-time_step:].reshape(1,-1)

temp_input=list(x_input)
temp_input=temp_input[0].tolist()

model = tf.keras.models.load_model('lstm.h5')


lst_output=[]
n_steps=time_step
i=0
dias=number
while(i<dias):
    
    if(len(temp_input)>n_steps):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        #print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        #print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        #print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        #print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1


time_step = 50
X_train, y_train = create_ds(train_data, time_step)
X_test, ytest = create_ds(test_data, time_step)
X_val,y_val = create_ds(validation_data, time_step)


train_predict=model.predict(X_train)
test_predict=model.predict(X_test)
val_predict=model.predict(X_val)


st.subheader('Resultado')
fig, ax = plt.subplots(figsize = (20,12))
ax.plot(df_data)
look_back=time_step

exp_step = time_step
exp_end = exp_step+len(train_predict)

trainPredictPlot = np.zeros((exp_end+1,1))
trainPredictPlot[:, :] = np.nan
trainPredictPlot[exp_step:exp_end, :] = train_predict
ax.plot(trainPredictPlot)

exp_step = exp_end+time_step+1
exp_end = exp_step+len(test_predict)

testPredictPlot = np.zeros((exp_end+1,1))
testPredictPlot[:, :] = np.nan
testPredictPlot[exp_step:exp_end, :] = test_predict
ax.plot(testPredictPlot)

exp_step = exp_end+time_step+1
exp_end = exp_step+len(val_predict)

valPredictPlot = np.zeros((exp_end+1,1))
valPredictPlot[:, :] = np.nan
valPredictPlot[exp_step:exp_end, :] = val_predict
ax.plot(valPredictPlot)


exp_step = len(train_predict)+len(test_predict)+(time_step*3)+2
exp_end = exp_step+len(lst_output)

expPredictPlot = np.zeros((exp_end+1,1))
expPredictPlot[:, :] = np.nan
expPredictPlot[exp_step:exp_end, :] = lst_output
ax.plot(expPredictPlot)

ax.set_title('Predict time')

ax.set_ylabel('value')
ax.set_xlabel('time')
ax.legend(['time serie', 'predict train','predict test','predict val','simulation with LSTM model'], loc='upper left')
st.pyplot(fig) # Display pyplot figure