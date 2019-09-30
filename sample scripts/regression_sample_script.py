from sklearn.preprocessing import LabelEncoder
from autosklearn.regression import AutoSklearnRegressor
from autosklearn.metrics import r2
import sklearn.model_selection
import numpy as np
import csv
import pandas as pd
import pickle
import json

#mostly {'No':0,'Yes':1}

df = pd.read_csv('../example_data_sets/wx_regress_dataset.csv', encoding = 'utf8')
print(df.head())

target_df = df['Rainfall']
inputs_df = df.drop('Rainfall', axis='columns')

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(inputs_df, target_df, random_state=1)
automl = AutoSklearnRegressor(time_left_for_this_task=3600) #10 minutes 600
automl.fit(X_train, y_train)
y_hat = automl.predict(X_test)
print("Accuracy score R2: ", r2(y_test, y_hat))

# TypeError: float() argument must be a string or a number, not 'Timestamp'
# df.ScheduledDay = df.ScheduledDay.apply(np.datetime64)
# df.AppointmentDay   = df.AppointmentDay.apply(np.datetime64)
# df['PatientId'] = df['PatientId'].astype(int)

# df['AppointmentDay'] = df['AppointmentDay'].dt.weekday
# df['ScheduledDay'] = df['ScheduledDay'].dt.weekday
# print(df.ScheduledDay.head())
# print(df.AppointmentDay.head())

# inputs = df.drop('NoShow', axis='columns')
# target = df['NoShow']

# le_target = LabelEncoder()
# target = le_target.fit_transform(target)
# le_target_mapping = dict(zip(le_target.classes_, le_target.transform(le_target.classes_).tolist()))
# with open('models_trained/noshow/NoShow_target_mapping.json', 'w', encoding='utf-8') as f:
# 	json.dump(le_target_mapping, f, indent=4)
# with open('models_trained/noshow/le_target.pkl', 'wb') as file:  
# 	pickle.dump(le_target, file)


# mapper = {}
# for col, col_data in inputs.iteritems():
# 	if col_data.dtype == object:
# 		le = LabelEncoder()
# 		inputs[str(col)] = le.fit_transform(inputs[str(col)])
# 		le_mapping = dict(zip(le.classes_, le.transform(le.classes_).tolist()))
# 		mapper[str(col)] = le_mapping

# 		with open('models_trained/noshow/{}.pkl'.format(str(col)), 'wb') as file:  
# 			pickle.dump(le_target, file) # --- CHECK THIS, It Should be le_ of columns

# with open('models_trained/noshow/NoShow_features_mapping.json', 'w', encoding='utf-8') as f:
# 	json.dump(mapper, f, indent=4)

# X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(inputs, target, random_state=1)
# automl = autosklearn.classification.AutoSklearnClassifier()
# automl.fit(X_train, y_train)
# y_hat = automl.predict(X_test)
# print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_hat))

# model_info = {
# 	'accuracy_score': str(sklearn.metrics.accuracy_score(y_test, y_hat)),
# 	'description': 'Predicting no show clients.'
# }
# with open('models_trained/noshow/info.json', 'w', encoding='utf-8') as f:
# 	json.dump(model_info, f, indent=4)

# with open('models_trained/noshow/noshow.pkl', 'wb') as f:
# 	pickle.dump(automl, f)





