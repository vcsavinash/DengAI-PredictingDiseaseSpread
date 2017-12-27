
# coding: utf-8

# In[688]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
input_dataset = pd.read_csv("C:/Users/Avinash/Documents/Third_Sem/MachineLearning/Project/dengue_features_train.csv")
input_dataset_label = pd.read_csv("C:/Users/Avinash/Documents/Third_Sem/MachineLearning/Project/dengue_labels_train.csv")
from sklearn.model_selection import train_test_split
input_dataset['station_avg_temp_c'] = input_dataset['station_avg_temp_c']+273.15
input_dataset['station_diur_temp_rng_c'] = input_dataset['station_diur_temp_rng_c']+273.15
input_dataset['station_max_temp_c'] = input_dataset['station_max_temp_c']+273.15
input_dataset['station_min_temp_c'] = input_dataset['station_min_temp_c']+273.15

#input_dataset.pop('week_start_date')
input_dataset = input_dataset.replace(r'\s+', np.nan, regex=True)

#input_dataset = input_dataset.fillna(X.mean(), inplace=True)
input_dataset.fillna(method='ffill', inplace=True)

merged_dataframes = [input_dataset, input_dataset_label['total_cases']]
merged_data = pd.concat(merged_dataframes, axis=1)

input_dataset_sj = merged_data.loc[merged_data['city'] == 'sj']
input_dataset_iq = merged_data.loc[merged_data['city'] == 'iq']


import datetime
from matplotlib.finance import date2num

dateList = []
temp = np.array(input_dataset_sj['week_start_date'])

for i in range(len(temp)):
    #print(  str(input_dataset_sj['week_start_date'].iloc[[i]])  )
    dateList.append(date2num(datetime.datetime.strptime( temp[i], "%m/%d/%Y"))/365)

plt.figure(figsize=(20,10)) 
plt.plot(dateList, input_dataset_sj['total_cases'])
plt.title("Total Cases for San Juan")
plt.xlabel("Weeks")
plt.ylabel("Total Cases")
plt.show()


# In[677]:

dateList = []
temp = np.array(input_dataset_iq['week_start_date'])

for i in range(len(temp)):
    #print(  str(input_dataset_sj['week_start_date'].iloc[[i]])  )
    dateList.append(date2num(datetime.datetime.strptime( temp[i], "%m/%d/%Y"))/365)
plt.figure(figsize=(20,10)) 
plt.plot(dateList, input_dataset_iq['total_cases'], color='red')
plt.title("Total Cases for Iquitos")
plt.xlabel("Weeks")
plt.ylabel("Total Cases")
plt.show()


# In[678]:

input_dataset_sj.pop('week_start_date')
input_dataset_iq.pop('week_start_date')
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.matshow(input_dataset_sj.corr(), fignum=1, cmap = 'hot')
plt.xticks(rotation=90)
plt.xticks(range(len(input_dataset_sj.columns)), input_dataset_sj.columns)
plt.yticks(range(len(input_dataset_sj.columns)), input_dataset_sj.columns)
plt.colorbar()
plt.show()


# In[679]:

import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.matshow(input_dataset_iq.corr(), fignum=1, cmap = 'hot')
plt.xticks(rotation=90)
plt.xticks(range(len(input_dataset_iq.columns)), input_dataset_iq.columns)
plt.yticks(range(len(input_dataset_iq.columns)), input_dataset_iq.columns)
plt.colorbar()
plt.show()


# In[680]:

results = {}
results["autocorr_sj"] = [pd.Series(input_dataset_sj['total_cases']).autocorr(lag) for lag in range(15)]
pd.DataFrame(results).plot(kind="bar")
plt.xlabel("lag")
plt.ylabel("Auto Correlation Function")
plt.show()


# In[681]:

results1 = {}
results1["autocorr_iq"] = [pd.Series(input_dataset_iq['total_cases']).autocorr(lag) for lag in range(15)]
pd.DataFrame(results1).plot(kind="bar")
plt.xlabel("lag")
plt.ylabel("Auto Correlation Function")
plt.show()


# In[682]:

pd.options.mode.chained_assignment = None
input_dataset_sj['lag1']=input_dataset_sj['total_cases'].shift(-1)
input_dataset_sj['lag2']=input_dataset_sj['lag1'].shift(-1)
input_dataset_sj['lag3']=input_dataset_sj['lag2'].shift(-1)

input_dataset_iq['lag1']=input_dataset_iq['total_cases'].shift(-1)
input_dataset_iq['lag2']=input_dataset_iq['lag1'].shift(-1)
input_dataset_iq['lag3']=input_dataset_iq['lag2'].shift(-1)


# In[683]:

input_dataset_sj


# In[662]:

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=2, random_state=0)
X_sj = input_dataset_sj[input_dataset_sj.columns[1:23]] #Selecting only features
y_sj = input_dataset_sj[input_dataset_sj.columns[23:24]] #Selecting total_cases
yt_sj = y_sj['total_cases']

clf.fit(X_sj, yt_sj)
importances_sj = clf.feature_importances_
indices_sj = np.argsort(importances_sj)
plt.title('Features Importance San Juan')
plt.barh(range(len(indices_sj)), importances_sj[indices_sj], color='b', align='center')
plt.yticks(range(len(indices_sj)), list(input_dataset_sj.columns.values))
plt.xlabel('Features Importance')
plt.show()


# In[638]:

clf = RandomForestClassifier(max_depth=2, random_state=0)
X_iq = input_dataset_iq[input_dataset_iq.columns[1:23]] #Selecting only features
y_iq = input_dataset_iq[input_dataset_iq.columns[23:24]] #Selecting total_cases
yt_iq = y_iq['total_cases']

clf.fit(X_iq, yt_iq)
importances_iq = clf.feature_importances_
indices_iq = np.argsort(importances_iq)
plt.title('Features Importance Iquitos')
plt.barh(range(len(indices_iq)), importances_iq[indices_iq], color='b', align='center')
plt.yticks(range(len(indices_iq)), list(input_dataset_iq.columns.values))
plt.xlabel("Features Importance")
plt.show()


# In[663]:

#,'reanalysis_tdtr_k','reanalysis_relative_humidity_percent','reanalysis_specific_humidity_g_per_kg','station_diur_temp_rng_c',reanalysis_avg_temp_k
removal_attributes_sj = ['ndvi_sw','ndvi_se','ndvi_nw','ndvi_ne','weekofyear','year', 'city', 'precipitation_amt_mm', 'reanalysis_tdtr_k', 
                         'reanalysis_sat_precip_amt_mm', 'reanalysis_dew_point_temp_k', 'reanalysis_max_air_temp_k']

refined_dataset_sj = input_dataset_sj.drop(removal_attributes_sj, axis=1)
removal_attributes_iq = ['ndvi_sw','ndvi_se','ndvi_nw','ndvi_ne','weekofyear','year', 'city', 'precipitation_amt_mm', 'reanalysis_air_temp_k', 'reanalysis_avg_temp_k']
refined_dataset_iq = input_dataset_iq.drop(removal_attributes_iq, axis=1)

from sklearn.model_selection import train_test_split
refined_dataset_sj = refined_dataset_sj.dropna()
refined_dataset_iq = refined_dataset_iq.dropna()

Y_sj = refined_dataset_sj.pop('total_cases')
X_sj = refined_dataset_sj
X_train_sj, X_test_sj, Y_train_sj, Y_test_sj = train_test_split(X_sj, Y_sj,test_size=0.20)

Y_iq = refined_dataset_iq.pop('total_cases')
X_iq = refined_dataset_iq
X_train_iq, X_test_iq, Y_train_iq, Y_test_iq = train_test_split(X_iq, Y_iq,test_size=0.20)


# In[664]:

from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR


clf_svm_sj = SVR(C=5.0, max_iter=631, epsilon=0.075)
clf_svm_sj.fit(X_train_sj, Y_train_sj)
prediction_svm_sj = clf_svm_sj.predict(X_test_sj)

clf_svm_iq = SVR(C=5.0, max_iter=631, epsilon=0.075)
clf_svm_iq.fit(X_train_iq, Y_train_iq)
prediction_svm_iq = clf_svm_iq.predict(X_test_iq)

mae = mean_absolute_error(Y_test_sj, prediction_svm_sj)
print("Mean Absolute Error values - MAE")
print ("SVR - San Juan: ", mae)


# In[665]:

mae = mean_absolute_error(Y_test_iq, prediction_svm_iq)
print ("SVR - Iquitos: ", mae)


# In[666]:

from sklearn.ensemble import GradientBoostingRegressor


clf_gb_sj = GradientBoostingRegressor(learning_rate=0.1, n_estimators=50, max_depth=3, max_leaf_nodes=None)
clf_gb_sj.fit(X_train_sj, Y_train_sj)
prediction_gb_sj = clf_gb_sj.predict(X_test_sj)

clf_gb_iq = GradientBoostingRegressor(learning_rate=0.1, n_estimators=50, max_depth=3, max_leaf_nodes=None)
clf_gb_iq.fit(X_train_iq, Y_train_iq)
prediction_gb_iq = clf_gb_iq.predict(X_test_iq)

mae = mean_absolute_error(Y_test_sj, prediction_gb_sj)
print ("Gradient Boost - San Juan: ", mae)


# In[667]:

mae = mean_absolute_error(Y_test_iq, prediction_gb_iq)
print ("Gradient Boost - Iquitos: ", mae)


# In[668]:

from sklearn.ensemble import AdaBoostRegressor


base_regressor = GradientBoostingRegressor()

clf_ab_sj = AdaBoostRegressor(base_estimator=base_regressor, n_estimators=25, learning_rate=0.5, loss='square')
clf_ab_sj.fit(X_train_sj, Y_train_sj)
prediction_ab_sj = clf_ab_sj.predict(X_test_sj)

clf_ab_iq = AdaBoostRegressor(base_estimator=base_regressor, n_estimators=25, learning_rate=0.5, loss='square')
clf_ab_iq.fit(X_train_iq, Y_train_iq)
prediction_ab_iq = clf_ab_iq.predict(X_test_iq)

mae = mean_absolute_error(Y_test_sj, prediction_ab_sj)
print ("AdaBoost - San Juan: ", mae)


# In[669]:

mae = mean_absolute_error(Y_test_iq, prediction_ab_iq)
print ("AdaBoost - Iquitos: ", mae)


# In[670]:

from sklearn.linear_model import LogisticRegression


clf_rl_sj = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=100)
clf_rl_sj.fit(X_train_sj, Y_train_sj)
prediction_rl_sj = clf_rl_sj.predict(X_test_sj)

clf_rl_iq = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=100)
clf_rl_iq.fit(X_train_iq, Y_train_iq)
prediction_rl_iq = clf_rl_iq.predict(X_test_iq)

mae = mean_absolute_error(Y_test_sj, prediction_rl_sj)
print ("Logistic Regression - San Juan: ", mae)


# In[671]:

mae = mean_absolute_error(Y_test_iq, prediction_rl_iq)
print ("Logistic Regression - Iquitos: ", mae)


# In[672]:

from sklearn.ensemble import RandomForestRegressor


clf_rf_sj = RandomForestRegressor(n_estimators=20, max_depth=7, bootstrap=True)
clf_rf_sj.fit(X_train_sj, Y_train_sj)
prediction_rf_sj = clf_rf_sj.predict(X_test_sj)

clf_rf_iq = RandomForestRegressor(n_estimators=20, max_depth=7, bootstrap=True)
clf_rf_iq.fit(X_train_iq, Y_train_iq)
prediction_rf_iq = clf_rf_iq.predict(X_test_iq)

mae = mean_absolute_error(Y_test_sj, prediction_rf_sj)
print ("Random Forest - San Juan: ", mae)


# In[673]:

mae = mean_absolute_error(Y_test_iq, prediction_rf_iq)
print ("Random Forest - Iquitos: ", mae)


# In[674]:

from sklearn.neural_network import MLPRegressor


clf_dl_sj = MLPRegressor(hidden_layer_sizes=(25, 25, 25, 25, 25), activation='relu',alpha=0.0001,learning_rate='adaptive',max_iter=500)
clf_dl_sj.fit(X_train_sj, Y_train_sj)
prediction_dl_sj = clf_dl_sj.predict(X_test_sj)

clf_dl_iq = MLPRegressor(hidden_layer_sizes=(25, 25, 25, 25, 25), activation='relu',alpha=0.0001,learning_rate='adaptive',max_iter=500)
clf_dl_iq.fit(X_train_iq, Y_train_iq)
prediction_dl_iq = clf_dl_iq.predict(X_test_iq)

mae = mean_absolute_error(Y_test_sj, prediction_dl_sj)
print ("Deep Learning - San Juan: ", mae)


# In[675]:

mae = mean_absolute_error(Y_test_iq, prediction_dl_iq)
print ("Deep Learning - Iquitos: ", mae)


# In[ ]:



