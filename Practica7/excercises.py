# https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
# https://mc.ai/understanding-input-and-output-shape-in-lstm-keras/

##############################
                 #
               # #
             #   #
                 #
                 #
##############################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set(style='whitegrid')

data=pd.read_csv('airline-passengers.csv')
fig=plt.figure(figsize=(9,6))
plt.plot(data['Passengers'])
xticks=range(len(data))
plt.xticks(xticks[::20], list(data['Month'].values)[::20], rotation='vertical')
plt.xlabel('Fecha', fontsize=14)
plt.ylabel('Pasajeros', fontsize=14)
plt.title('Data set de  airlines passengers', fontsize=16)
plt.savefig('plot_data.pdf', format='pdf')
plt.show()
x=data['Passengers'].values

##############################
            ######
                 #
            ######
            # 
            ######      
##############################
data_min=np.min(x)
data_max=np.max(x)
x_minmax=((x-data_min)/(data_max-data_min)).reshape(-1,1)

def create_x_y(x, lag=1):
    y=[]
    x_new=[]
    for i in range(len(x)-lag):
        x_new.append(x[i:i+lag])
        y.append(x[i+lag])
    return np.array(x_new).reshape(len(x)-lag, lag), np.array(y).reshape(-1)

lag=1
x, y=create_x_y(x_minmax, lag)

##############################
            ######
                 #
            ######
                 #
            ######      
##############################
y+=np.random.normal(0, 0.02, size=y.shape)

##############################
            #    #
            #    #
            ######
                 #
                 #      
##############################
def split_train_test(x, y, ratio=0.5):
    test_size=int(ratio*x.shape[0])
    train_size=x.shape[0]-test_size
    x_train=x[:train_size]
    x_test=x[train_size:]
    y_train=y[:train_size]
    y_test=y[train_size:]
    return x_train, x_test, y_train, y_test, train_size, test_size

x_train, x_test, y_train, y_test, train_size, test_size = split_train_test(x, y, 0.5)

##############################
            ######
            #     
            ######
                 #
            ######      
##############################
x_train=x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test=x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

##############################
            ######
            #     
            ######
            #    #
            ######      
##############################
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

lr=1e-3

model = Sequential(name='red_recurrente_LSTM')
model.add(LSTM(16, 
               input_shape=(lag, 1), 
               name='LSTM'))
model.add(Dense(1,
                name='Dense'))
model.compile(loss=keras.losses.MeanSquaredError(), 
            optimizer=keras.optimizers.Adam(lr))

model.summary()

##############################
            ######
                 #
                #  
               #
               #  
##############################
epochs=200
batch_size=10

history=model.fit(x_train,
          y_train, 
          epochs=epochs, 
          batch_size=batch_size, 
          verbose=2)

fig=plt.figure(figsize=(9,6))
ax1 = plt.gca()
ax1.tick_params(axis = 'both', which = 'major', labelsize = 14)
ax1=plt.subplot()
#ax1.scatter(np.arange(epochs), history.history['loss'], color='red', alpha=0.5, marker='^', s=10)
ax1.plot(np.arange(epochs), history.history['loss'], color='red', label='Training loss')
ax1.set_xlabel('Epoca', fontsize=14), ax1.set_ylabel('Costo', fontsize=14)
ax1.hlines(min(history.history['loss']), 0, epochs, color='red', linestyles='dashdot', alpha=0.5)
ax1.set_yscale('log')
plt.title('Resultados de entrenamiento para EMNIST. Lr='+str(lr)+', rf='+str(0), fontsize=15)
ax1.legend()
plt.savefig('ej7.pdf', format='pdf')
plt.show()
plt.close()


##############################
            ######
            #    #
            ######  
            #    #
            ######  
##############################
y_train_pred = model.predict(x_train).reshape(-1, 1)
y_test_pred = model.predict(x_test).reshape(-1, 1)
loss_train = model.evaluate(x_train, y_train)
loss_test = model.evaluate(x_test, y_test)

y_train_pred = y_train_pred*(data_max-data_min)+data_min
y_test_pred = y_test_pred*(data_max-data_min)+data_min

fig=plt.figure(figsize=(9,6))
plt.plot(data['Passengers'], label='Datos originales')
plt.plot(lag-1+np.arange(0, train_size, 1), y_train_pred, label='Prediccion sobre los datos de training')
plt.plot(lag-1+train_size+np.arange(0, test_size, 1), y_test_pred, label='Prediccion sobre los datos de test')
xticks=range(len(data))
plt.xticks(xticks[::20], list(data['Month'].values)[::20], rotation='vertical')
plt.xlabel('Fecha', fontsize=14)
plt.ylabel('Pasajeros', fontsize=14)
plt.title('Data set de  airlines passengers', fontsize=16)
plt.legend()
plt.savefig('ej8.pdf', format='pdf')
plt.show()

##############################
            ######
            #    #
            ######  
                 #
                 #  
##############################
epochs=200
batch_size=10
fig = plt.figure(figsize=(9,6))
loss_plot=[]
for i in range(20):
    lag=i+1
    x=data['Passengers'].values
    x_minmax=((x-data_min)/(data_max-data_min)).reshape(-1,1)
    x, y=create_x_y(x_minmax, lag)
    x_train, x_test, y_train, y_test, train_size, test_size = split_train_test(x, y, 0.5)
    x_train=x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test=x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
        
    lr=1e-3

    model = Sequential(name='red_recurrente_LSTM')
    model.add(LSTM(16, 
              input_shape=(lag, 1), 
              name='LSTM'))
    model.add(Dense(1, name='Dense'))
    model.compile(loss=keras.losses.MeanSquaredError(), 
                optimizer=keras.optimizers.Adam(lr))
    model.fit(x_train,
          y_train, 
          epochs=epochs, 
          batch_size=batch_size,
          verbose=0)
    loss=model.evaluate(x_test, y_test)
    loss_plot.append(loss)
    y_train_pred = model.predict(x_train).reshape(-1, 1)

y_train_pred = y_train_pred*(data_max-data_min)+data_min
y_test_pred = y_test_pred*(data_max-data_min)+data_min
plt.scatter(range(20), loss_plot, color='C0')
plt.plot(range(20), loss_plot, color='C0')
plt.plot()
plt.title('Loss en función de la retrospección', fontsize=16)
plt.xlabel('Retrospección')
plt.ylabel('Loss')
plt.savefig('ej9_loss.pdf', format='pdf')
plt.show()

##############################
          #   ######
          #   #    #
          #   #    #  
          #   #    #
          #   ######  
##############################
x=data['Passengers'].values
data_min=np.min(x)
data_max=np.max(x)
x_minmax=((x-data_min)/(data_max-data_min)).reshape(-1,1)

def create_x_y(x, lag=1):
    y=[]
    for i in range(len(x)-lag):
        y.append(x[i+lag])
    return x[:-lag], np.array(y)

lag=1
x, y=create_x_y(x_minmax, lag)

y+=np.random.normal(0, 0.02, size=y.shape)
def split_train_test(x, y, ratio=0.5):
    test_size=int(ratio*x.shape[0])
    train_size=x.shape[0]-test_size
    x_train=x[:train_size]
    x_test=x[test_size:]
    y_train=y[:train_size]
    y_test=y[test_size:]
    return x_train, x_test, y_train, y_test, train_size, test_size

x_train, x_test, y_train, y_test, train_size, test_size = split_train_test(x, y, 0.5)

lr=1e-3
rf=0

model = Sequential(name='red_recurrente_Dense')
model.add(Dense(16, 
               input_shape=(x_train.shape[1],),
               activation='relu',
               kernel_regularizer=keras.regularizers.l2(rf), 
               name='Dense_1'))
model.add(Dense(1,
                name='Dense_2'))
model.compile(loss=keras.losses.MeanSquaredError(), 
            optimizer=keras.optimizers.Adam(lr))
model.summary()

epochs=200
batch_size=10

history=model.fit(x_train,
          y_train, 
          epochs=epochs, 
          batch_size=batch_size, 
          verbose=2)


fig=plt.figure(figsize=(9,6))
ax1 = plt.gca()
ax1.tick_params(axis = 'both', which = 'major', labelsize = 14)
ax1=plt.subplot()
#ax1.scatter(np.arange(epochs), history.history['loss'], color='red', alpha=0.5, marker='^')
ax1.plot(np.arange(epochs), history.history['loss'], color='red', label='Training loss')
ax1.set_xlabel('Epoca', fontsize=14), ax1.set_ylabel('Costo', fontsize=14)
ax1.hlines(min(history.history['loss']), 0, epochs, color='red', linestyles='dashdot', alpha=0.5)
ax1.set_yscale('log')
plt.title('Resultados de entrenamiento para EMNIST. Lr='+str(lr)+', rf='+str(0), fontsize=15)
ax1.legend()
plt.savefig('ej10_a.pdf', format='pdf')
plt.show()
plt.close()


y_train_pred = model.predict(x_train).reshape(-1, 1)
y_test_pred = model.predict(x_test).reshape(-1, 1)
loss_train = model.evaluate(x_train, y_train)
loss_test = model.evaluate(x_test, y_test)

y_train_pred = y_train_pred*(data_max-data_min)+data_min
y_test_pred = y_test_pred*(data_max-data_min)+data_min

fig=plt.figure(figsize=(9,6))
plt.plot(data['Passengers'], label='Datos originales')
plt.plot(np.arange(0, train_size, 1), y_train_pred, label='Prediccion sobre los datos de training')
if train_size!=test_size:
    plt.plot(train_size+np.arange(0, test_size+1, 1), y_test_pred, label='Prediccion sobre los datos de test')
else:
    plt.plot(train_size+np.arange(0, test_size, 1), y_test_pred, label='Prediccion sobre los datos de test')
plt.xlabel('Fecha', fontsize=14)
xticks=range(len(data))
plt.xticks(xticks[::20], list(data['Month'].values)[::20], rotation='vertical')
plt.ylabel('Pasajeros', fontsize=14)
plt.title('Data set de  airlines passengers', fontsize=16)
plt.legend()
plt.savefig('ej10_b.pdf', format='pdf')
plt.show()