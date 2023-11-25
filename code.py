# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 20:55:20 2023

@author: ericj
"""



# Spacecraft Anomaly Detection 


import os
import datetime
from math import sqrt
import pandas as pd
from pandas import datetime
import numpy as np
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMAResults
from sklearn.metrics import mean_squared_error
import sklearn.svm as svm
import matplotlib.pyplot as plt
import math

df=pd.read_csv('./WheelTemperature.csv')
df_battemp=pd.read_csv('./BatteryTemperature.csv')
df_buscurrent=pd.read_csv('./TotalBusCurrent.csv')
df_busvolt=pd.read_csv('./BusVoltage.csv')

df_battemp.Date = pd.to_datetime(df_battemp.Date, format="%m/%d/%Y %H:%M")
df_buscurrent.Date = pd.to_datetime(df_buscurrent.Date, format="%m/%d/%Y")
df_busvolt.Date=pd.to_datetime(df_busvolt.Date, format="%m/%d/%Y %H:%M")
df.Date = pd.to_datetime(df.Date, format="%m/%d/%Y %H:%M")

df_battemp=df_battemp.resample('1D',on='Date').mean()
df_buscurrent=df_buscurrent.resample('1D',on='Date').mean()
df_busvolt=df_busvolt.resample('1D',on='Date').mean()
df_busvolt=df_busvolt.loc['2004-02-13':]
df=df.resample('1D',on='Date').mean()

df['DATE2']=df.index
df['DATE2']=(pd.to_numeric(df['DATE2'])-1076677200000000000)/(461847000000000000/2)


from sklearn.cluster import DBSCAN
dbscan=DBSCAN(eps=0.3)
dbscan.fit(df)

colors = dbscan.labels_
outliers=colors.T<0
normal=colors.T>=0
print("Number of outliers detected: %d" % sum(i<0 for i in colors.T))
print("Number of normal samples detected: %d" % sum(i>=0 for i in colors.T))



fig, ax = plt.subplots(figsize=(9,9))
plt.plot(df.High[colors==0],marker='o',linestyle="None")
plt.plot(df.High[colors!=0],marker='o',linestyle="None")
plt.legend(["Valid Data","Data marked as outliers"])
plt.ylabel("Temperature (C)")
#plt.xlabel("Mean Radius")
plt.title("Dataset Outlier Detection via DBSCAN")
plt.show()

## this shows outlier points of all data through DBSCAN
## now generate forecast with ARIMA




#%% ARIMA
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error

df=pd.concat([df,df_battemp,df_buscurrent,df_busvolt],axis=1)
lag_features = ["High", "Low", "Volume", "Turnover", "NumTrades"]
lag_features=["High","Temp","Current","Voltage"]
window1 = 3
window2 = 7
window3 = 30

df_rolled_3d = df[lag_features].rolling(window=window1, min_periods=0)
df_rolled_7d = df[lag_features].rolling(window=window2, min_periods=0)
df_rolled_30d = df[lag_features].rolling(window=window3, min_periods=0)

df_mean_3d = df_rolled_3d.mean().shift(1).reset_index()
df_mean_7d = df_rolled_7d.mean().shift(1).reset_index()
df_mean_30d = df_rolled_30d.mean().shift(1).reset_index()

df_std_3d = df_rolled_3d.std().shift(1).reset_index()
df_std_7d = df_rolled_7d.std().shift(1).reset_index()
df_std_30d = df_rolled_30d.std().shift(1).reset_index()

df_mean_3d.set_index("Date", drop=True, inplace=True)
df_mean_7d.set_index("Date", drop=True, inplace=True)
df_mean_30d.set_index("Date", drop=True, inplace=True)
df_std_3d.set_index("Date", drop=True, inplace=True)
df_std_7d.set_index("Date", drop=True, inplace=True)
df_std_30d.set_index("Date", drop=True, inplace=True)

for feature in lag_features:
    
    df[f"{feature}_mean_lag{window1}"] = df_mean_3d[feature]
    df[f"{feature}_mean_lag{window2}"] = df_mean_7d[feature]
    df[f"{feature}_mean_lag{window3}"] = df_mean_30d[feature]
    
    df[f"{feature}_std_lag{window1}"] = df_std_3d[feature]
    df[f"{feature}_std_lag{window2}"] = df_std_7d[feature]
    df[f"{feature}_std_lag{window3}"] = df_std_30d[feature]


df.fillna(df.mean(), inplace=True)

data=df
df_train=data.iloc[0:math.floor(len(data)*.75),:]
df_valid=data.iloc[math.floor(len(data)*.75):,:]
exogenous_features=['High_mean_lag3', 'High_mean_lag7',
       'High_mean_lag30', 'High_std_lag3', 'High_std_lag7', 'High_std_lag30',
       'Temp_mean_lag3', 'Temp_mean_lag7', 'Temp_mean_lag30', 'Temp_std_lag3',
       'Temp_std_lag7', 'Temp_std_lag30', 'Current_mean_lag3',
       'Current_mean_lag7', 'Current_mean_lag30', 'Current_std_lag3',
       'Current_std_lag7', 'Current_std_lag30', 'Voltage_mean_lag3',
       'Voltage_mean_lag7', 'Voltage_mean_lag30', 'Voltage_std_lag3',
       'Voltage_std_lag7', 'Voltage_std_lag30']
model = auto_arima(
    df_train["High"],
    exogenous=df_train[exogenous_features],
    trace=True,
    error_action="ignore",
    suppress_warnings=True,
    seasonal=True,
    m=1)

model.fit(df_train.High, exogenous=df_train[exogenous_features])
forecast = model.predict(n_periods=len(df_valid), exogenous=df_valid[exogenous_features])
df_valid.insert(len(df_valid.columns),"Forecast_ARIMAX",forecast,True)


df_valid[["High", "Forecast_ARIMAX"]].plot(figsize=(9, 5))
plt.legend(['Wheel Temperature (Truth)','Forecast (ARIMA)'])
plt.show()




dd=df_valid[["Forecast_ARIMAX"]]
dd['DATE2']=df_valid.index
dd['DATE2']=(pd.to_numeric(dd['DATE2'])-1076677200000000000)/(461847000000000000/2)

dbscan=DBSCAN(eps=0.4)
dbscan.fit(dd)

colors = dbscan.labels_
outliers=colors.T<0
normal=colors.T>=0
print("Number of outliers detected: %d" % sum(i<0 for i in colors.T))
print("Number of normal samples detected: %d" % sum(i>=0 for i in colors.T))



fig, ax = plt.subplots(figsize=(9,9))
plt.plot(dd.Forecast_ARIMAX[colors==0],marker='o',linestyle="None")
plt.plot(dd.Forecast_ARIMAX[colors!=0],marker='o',linestyle="None")
plt.legend(["Valid Data","Data marked as outliers"])
plt.ylabel("Temperature (C)")
#plt.xlabel("Mean Radius")
plt.title("Dataset Outlier Detection via DBSCAN")
plt.show()




# try using dbscan or ocsvm on full set of data
# PCA might be able to decompose 
# https://www.kaggle.com/code/kevinarvai/outlier-detection-practice-uni-multivariate


#%%
from sklearn.decomposition import PCA

pca = PCA(n_components=1)
xt=pca.fit(df).transform(df)

X_reconstructed_pca = pca.inverse_transform(pca.transform(df))

fig, ax = plt.subplots(figsize=(9,9))


outliers_fraction = 0.01
from sklearn.ensemble import IsolationForest
ifo = IsolationForest(contamination = outliers_fraction)

#xt=np.array(df.High.values).reshape(-1,1)
ifo.fit(xt)
anom1=pd.Series(ifo.predict(xt))
a=anom1[anom1==-1]
plt.plot(df.High.values)
plt.scatter(a.index,df.High.values[a.index],c='Red')






#%%
fig, ax = plt.subplots(4,1,figsize=(9,9))
plt.subplot(411)
plt.plot(df.High)
plt.subplot(412)
plt.plot(df.Temp)
plt.subplot(413)
plt.plot(df.Current)
plt.subplot(414)
plt.plot(df.Voltage)



#%% EPOXI dataset, 2 sensors on Deep Impact
df=pd.read_csv('./test.csv')
df['Date'] = pd.to_datetime(df.SCET, format="%Y-%m-%dT%H:%M:%S.%f")
df.drop(['SCET'],axis=1)
df=df.resample('60T',on='Date').mean()
#df=df[['I_0221','I_1249','I_0342']]
df=df[df.keys()[[0,4,6,8,10,12,14,16,18,20,22,24,26,28,30,31]]]


df_train=df.iloc[0:math.floor(len(df)*.6),:]
df_valid=df.iloc[math.floor(len(df)*.6):,:]

#df=df_train
#df_valid=df
#df_train=df

pca = PCA(n_components=2)
#pca = PCA()
pca.fit(df_train).transform(df_valid)
features = range(pca.n_components_)
_=plt.figure(figsize=(22,5))
_=plt.bar(features,pca.explained_variance_)
_=plt.xlabel('PCA feature')
_=plt.ylabel('Variance')
_=plt.xticks(features)
_=plt.title('Important Principal Component')
plt.show()


xt=pca.fit(df_train).transform(df_valid)
xt_train=pca.fit(df_train).transform(df_train)
xt_valid=pca.fit(df_valid).transform(df_valid)

outliers_fraction = 0.01
from sklearn.ensemble import IsolationForest
ifo = IsolationForest(contamination=outliers_fraction)

#xt=np.array(df.High.values).reshape(-1,1)
ifo.fit(xt_train)
anom1=pd.Series(ifo.predict(xt_valid))
a=anom1[anom1==-1]



model =svm.OneClassSVM(nu=.005)
model.fit(xt_train)
#anom=(pd.Series(model.predict(xt_valid)))
#a=anom[anom==-1]

fig, ax = plt.subplots(figsize=(9,7))
#plt.plot(df_train.I_0233)
plt.plot(df_valid.I_0233)
plt.scatter(df_valid.index[a.index],df_valid.I_0233.values[a.index],c='Red')




# KMEANS

def getDistanceByPoint(data, model):
    distance = pd.Series()
    for i in range(0,len(data)):
        Xa = np.array(data.loc[i])
        Xb = model.cluster_centers_[model.labels_[i]-1]
        distance.at[i] = np.linalg.norm(Xa-Xb)
    return distance

from sklearn.cluster import KMeans
n_cluster = range(1, 20)
kmeans = [KMeans(n_clusters = i).fit(xt) for i in n_cluster]
clus=kmeans[7].predict(xt)
pc1=xt[:,0];
pc2=xt[:,1];


outliers_fraction = 0.01
data=pd.DataFrame(xt)
distance = getDistanceByPoint(data, kmeans[9])
outlier_num = int(outliers_fraction * len(distance))
threshold = distance.nlargest(outlier_num).min()
an=(distance >= threshold).astype(int)
a=an[an==1];

fig, ax = plt.subplots(figsize = (12, 6))

colors = {0:'lightgreen', 1:'red'}

ax.scatter(pc1,pc2, 
           c = an.apply(lambda x: colors[x]))

plt.xlabel('pc1')
plt.ylabel('pc2')
plt.show();

plt.plot(df_valid.I_0233)
plt.scatter(df_valid.index[a.index],df_valid.I_0233.values[a.index],c='Red')




#%% LSTM
df=pd.read_csv('./test.csv')
df['Date'] = pd.to_datetime(df.SCET, format="%Y-%m-%dT%H:%M:%S.%f")
df.drop(['SCET'],axis=1)
df=df.resample('60T',on='Date').mean()
#df=df[df.keys()[[0,4,6,8,10,12,14,16,18,20,22,24,26,28,30,31]]]

df_train=df.iloc[0:math.floor(len(df)*.8),:]
df_valid=df.iloc[math.floor(len(df)*.8):,:]
pca = PCA(n_components=1)
#pca = PCA()
#pca.fit(df_train).transform(df_valid)
features = range(pca.n_components)
xt=pca.fit(df).transform(df)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
#dataset = scaler.fit_transform(xt[:,1].reshape(-1,1))
dataset = scaler.fit_transform(xt[:,0].reshape(-1,1))
train_size = int(len(dataset) * .8)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))



def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

t=pd.DataFrame(train)
t['t']=train
dataset=t.values

model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(train, train, epochs=10, batch_size=1, verbose=2)


trainPredict = model.predict(train)
testPredict = model.predict(test)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[2:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = np.sqrt(mean_squared_error(testY[0], testPredict[2:,0]))
print('Test Score: %.2f RMSE' % (testScore))


from sklearn.ensemble import IsolationForest
ifo = IsolationForest(contamination=outliers_fraction)

#xt=np.array(df.High.values).reshape(-1,1)
ifo.fit(testPredict)
anom1=pd.Series(ifo.predict(testPredict))
a=anom1[anom1==-1]
model =svm.OneClassSVM(nu=.005)
model.fit(testPredict)
#anom=(pd.Series(model.predict(xt_valid)))
#a=anom[anom==-1]
fig, ax = plt.subplots(figsize=(9,7))
plt.plot(df_train.I_1251)
plt.plot(df_valid.I_1251)
plt.scatter(df_valid.index[a.index],df_valid.I_1251.values[a.index],c='Red')









#%%
import tensorflow as tf 
from keras.layers import Input, Dense 
from keras.models import Model 
from sklearn.metrics import precision_recall_fscore_support 




data = pd.read_csv( 
    'https://raw.githubusercontent.com/numenta'
    '/NAB/master/data/realKnownCause/ambient'
    '_temperature_system_failure.csv') 
df=pd.read_csv('./test.csv')
df['Date'] = pd.to_datetime(df.SCET, format="%Y-%m-%dT%H:%M:%S.%f")
df.drop(['SCET'],axis=1)
df=df.resample('.25D',on='Date').mean()
#df=pd.DataFrame(df['I_0233'].values,columns=['I_0233'])
data_tensor = tf.convert_to_tensor(df.values, dtype=tf.float32) 
input_dim = df.shape[1]
encoding_dim = 10
input_layer = Input(shape=(input_dim,)) 
encoder = Dense(encoding_dim, activation='relu')(input_layer) 
decoder = Dense(input_dim, activation='relu')(encoder) 
autoencoder = Model(inputs=input_layer, outputs=decoder) 
autoencoder.compile(optimizer='adam', loss='mse') 
autoencoder.fit(data_tensor, data_tensor, epochs=50, 
                batch_size=32, shuffle=True) 

reconstructions = autoencoder.predict(data_tensor) 
mse = tf.reduce_mean(tf.square(data_tensor - reconstructions), 
                     axis=1) 
anomaly_scores = pd.Series(mse.numpy(), name='anomaly_scores') 
anomaly_scores.index = df.index 


threshold = anomaly_scores.quantile(0.99) 
anomalous = anomaly_scores > threshold 
binary_labels = anomalous.astype(int) 
precision, recall,    f1_score, _ = precision_recall_fscore_support( 
        binary_labels, anomalous, average='binary') 
    
    
test = df['I_0233'].values 
predictions = anomaly_scores.values 
  
print("Precision: ", precision) 
print("Recall: ", recall) 
print("F1 Score: ", f1_score) 




plt.figure(figsize=(16, 8)) 
plt.plot(df['I_0233']) 
plt.plot(df['I_0233'][anomalous], 'ro') 
plt.title('Anomaly Detection') 
plt.xlabel('Time') 
plt.ylabel('Value') 
plt.show() 




from tensorflow.keras import layers, losses
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
class detector(Model):
  def __init__(self, latent_dim, shape):
    super(detector, self).__init__()
    self.latent_dim = latent_dim
    self.shape = shape
    self.encoder = tf.keras.Sequential([
      layers.Flatten(),
      layers.Dense(latent_dim, activation='relu'),
    ])
    self.decoder = tf.keras.Sequential([
      layers.Dense(tf.math.reduce_prod(shape), activation='sigmoid'),
      layers.Reshape(shape)
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

latent_dim = 64
shape=data.shape[1:]
autoencoder = detector(latent_dim,shape)
autoencoder.compile(optimizer='adam', loss='mae')
def plot(data, n):
  enc_img = autoencoder.encoder(data)
  dec_img = autoencoder.decoder(enc_img)
  plt.plot(data[n], 'b')
  plt.plot(dec_img[n], 'r')
  plt.fill_between(np.arange(len(dec_img[n])), data[n], dec_img[n], color = 'lightcoral')
  plt.legend(labels=['Input', 'Reconstruction', 'Error'])
  plt.show()

plot(df['I_0233'], 0)
plot(an_test_data, 0)







