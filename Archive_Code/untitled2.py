# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 17:39:53 2023

@author: ericj
"""
pp=pd.DataFrame(columns=['FPR','nu'])
for i in range(250,350,5):
    st=i/10000
    print(st)
    outliers_fraction = st
    from sklearn.ensemble import IsolationForest
    ifo = IsolationForest(contamination=outliers_fraction)
    ifo.fit(xt_train)
    anom1=pd.Series(ifo.predict(xt_valid))
    a=anom1[anom1==-1]
    anom_ocsvm=anom1

    df_truth=truth
    anom=anom_ocsvm.map(lambda val:1 if val==-1 else 0)
    #calculate F1 score
    f1=f1_score((df_truth['Truth_anom'].values),(anom.values))
    rec=recall_score(df_truth['Truth_anom'].values, anom.values)
    prec=precision_score(df_truth['Truth_anom'].values, anom.values)
    TP, FP, TN, FN=perf_measure(df_truth['Truth_anom'].values, anom.values)
    fpr=FP/(TN+FP)
    
    print('FPR: %3.3f, nu = %3.3f' % (fpr,st))
    pp2=pd.DataFrame([[fpr,st]],columns=['FPR','nu'])
    pp=pp.append(pp2)
    plt.plot(pp.nu,pp.FPR)
    plt.show()
    
    
    
    
    
    
    
df=pd.read_csv('./test.csv')
df['Date'] = pd.to_datetime(df.SCET, format="%Y-%m-%dT%H:%M:%S.%f")
df.drop(['SCET'],axis=1)
df=df.resample('60T',on='Date').mean()
#df=df[['I_0221','I_1249','I_0342']]
df=df[df.keys()[[0,2,6,8,10,12,14,16,18,20,21,22,24,26,28,30]]]
#df=df[df.keys()[[0,2,4,6,8]]]


df_train=df.iloc[0:math.floor(len(df)*.6),:]
df_valid=df.iloc[math.floor(len(df)*.6):,:]

#df=df_train
#df_valid=df
#df_train=df

pca = PCA(n_components=16)
#pca = PCA()
pca.fit(df_train).transform(df_valid)
features = range(pca.n_components_)


xt=pca.fit(df_train).transform(df_valid)
xt_train=pca.fit(df_train).transform(df_train)
xt_valid=pca.fit(df_valid).transform(df_valid)

outliers_fraction = 0.0305
from sklearn.ensemble import IsolationForest
ifo = IsolationForest(contamination=outliers_fraction)

#xt=np.array(df.High.values).reshape(-1,1)
ifo.fit(xt_train)
anom1=pd.Series(ifo.predict(xt_valid))
a=anom1[anom1==-1]



model =svm.OneClassSVM(nu=.16)
model.fit(xt_train)
anom_ocsvm=(pd.Series(model.predict(xt_valid)))
a_ocsvm=anom_ocsvm[anom_ocsvm==-1]

fig, ax = plt.subplots(2,1,figsize=(12,9))
plt.subplot(211)
plt.plot(df_valid.I_0221)
plt.scatter(df_valid.index[a.index],df_valid.I_0221.values[a.index],c='Red')
plt.title('Isolation Forest Detected Anomalies')
#fig, ax = plt.subplots(figsize=(9,7))
plt.subplot(212)
plt.plot(df_valid.I_0221)
plt.scatter(df_valid.index[a_ocsvm.index],df_valid.I_0221.values[a_ocsvm.index],c='Red')
plt.title('OCSVM Detected Anomalies (nu = %s)' % model.get_params()['nu'])


df_truth=truth
anom=anom1.map(lambda val:1 if val==-1 else 0)
#calculate F1 score
f1=f1_score((df_truth['Truth_anom'].values),(anom.values))
rec=recall_score(df_truth['Truth_anom'].values, anom.values)
prec=precision_score(df_truth['Truth_anom'].values, anom.values)
TP, FP, TN, FN=perf_measure(df_truth['Truth_anom'].values, anom.values)
fpr=FP/(TN+FP)

print('f1: %3.6f\nrecall: %3.6f\nprecision: %3.6f\nfpr: %3.6f\n' %(f1,rec,prec,fpr))




df_truth=truth
anom=anom_ocsvm.map(lambda val:1 if val==-1 else 0)
#calculate F1 score
f1=f1_score((df_truth['Truth_anom'].values),(anom.values))
rec=recall_score(df_truth['Truth_anom'].values, anom.values)
prec=precision_score(df_truth['Truth_anom'].values, anom.values)
TP, FP, TN, FN=perf_measure(df_truth['Truth_anom'].values, anom.values)
fpr=FP/(TN+FP)

print('f1: %3.6f\nrecall: %3.6f\nprecision: %3.6f\nfpr: %3.6f\n' %(f1,rec,prec,fpr))
