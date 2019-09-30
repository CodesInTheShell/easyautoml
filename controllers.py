import csv
import pandas
import json
from sklearn.preprocessing import LabelEncoder
import autosklearn.classification
import sklearn.model_selection
import helpers
import config
import pickle
import os
from flask import session, request
import numpy as np
import datetime


class TrainStartManager():

	def start_training(self, targetname, df, modelname, model_desc, traning_time):

		status = {}

		self.modelname = modelname
		self.df = df 
		self.target_df = df[targetname]
		self.inputs_df = self.df.drop(targetname, axis='columns')
		self.model_desc = model_desc
		self.traning_time = traning_time

		# self.target_df, self.inputs_df = automltk.Utils().prep_df(targetname, self.df)

		# le_target_mapping = automltk.Utils().label_encode_target(self.target_df)

		proj_dir = helpers.create_project_dir(self.modelname)

		if proj_dir == False:
			status['status'] = 'Error'
			status['message'] = 'Model already exist'
			return status

		proj_dir = proj_dir+'/'

		le_target = LabelEncoder()
		self.target_df = le_target.fit_transform(self.target_df)
		le_target_mapping = dict(zip(le_target.classes_, le_target.transform(le_target.classes_).tolist()))
		with open(proj_dir + 'target_mapping.json', 'w', encoding='utf-8') as f:
			json.dump(le_target_mapping, f, indent=4)

		with open(proj_dir + 'le_target.pkl', 'wb') as file:  
			pickle.dump(le_target, file)

		# le_features_mapping = automltk.Utils().label_encode_freatures(self.inputs_df)
		# with open(proj_dir + 'features_mapping.json', 'w', encoding='utf-8') as f:
		# 	json.dump(le_features_mapping, f, indent=4)

		mapper = {}
		cols = []
		for col, col_data in self.inputs_df.iteritems():
			cols.append(str(col))
			if col_data.dtype == object:
				le = LabelEncoder()
				self.inputs_df[str(col)] = le.fit_transform(self.inputs_df[str(col)])
				le_mapping = dict(zip(le.classes_, le.transform(le.classes_).tolist()))
				mapper[str(col)] = le_mapping

				with open(proj_dir + '{}.pkl'.format(str(col)), 'wb') as file:  
					pickle.dump(le, file)

		with open(proj_dir + 'features_mapping.json', 'w', encoding='utf-8') as f:
			json.dump(mapper, f, indent=4)

		with open(proj_dir + 'columns.txt', 'w') as col_f:
			for c in cols:
				col_f.write("%s\n" % c)

		X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(self.inputs_df, self.target_df, random_state=1)
		automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=self.traning_time)
		automl.fit(X_train, y_train)
		y_hat = automl.predict(X_test)
		accuracy = str(sklearn.metrics.accuracy_score(y_test, y_hat))
		print("Accuracy score", accuracy)

		model_info = {
			'accuracy_score': accuracy,
			'description': str(self.model_desc)
		}
		with open(proj_dir + 'model_info.json', 'w', encoding='utf-8') as f:
			json.dump(model_info, f, indent=4)

		with open(proj_dir + '{}.pkl'.format(self.modelname), 'wb') as f:
			pickle.dump(automl, f)

		status['status'] = 'Success'
		status['message'] = 'Model has been created'
		status['accuracy'] = accuracy
		status['modelname'] =self.modelname
		return status


class ModelsLoader():

	def load_all_models(self):
		models =  [name for name in os.listdir(config.MODELS_DIR)]
		return models


class SessionManager():

	def add_to_session(self, session, key, value):
		session[key] = value 
		return session

class PredictStartManager():

	def start_predict():

		features_mappings_json = config.MODELS_DIR + session['modelselected'] + '/features_mapping.json'
		with open(features_mappings_json) as json_file:
			mappings = json.load(json_file)

		predictors = []
		cols = session['model_cols']
		for c in cols:
			if c in mappings.keys():
				predictors.append(mappings[c][request.form[c]])
			else:
				predictors.append(int(request.form[c]))

		new_data = np.array([predictors])

		with open(config.MODELS_DIR + session['modelselected'] + '/' + session['modelselected'] + '.pkl', 'rb') as model_pkl:
			model = pickle.load(model_pkl)
		with open(config.MODELS_DIR + session['modelselected'] + '/' + 'le_target.pkl', 'rb') as target_pkl:
			le_target = pickle.load(target_pkl)

		class_predicted = str(list(le_target.inverse_transform(model.predict(new_data)))[0])
		classes_proba = dict(zip(le_target.classes_, model.predict_proba(new_data)[0]))

		print('========================================================================')
		print('class_predicted: ', class_predicted)
		print('classes_proba: ', classes_proba)

		predicted_proba_value = classes_proba[class_predicted]

		pred_details = {
		'class_predicted': class_predicted,
		'predicted_proba_value': round(float(predicted_proba_value) * 100)
		}

		session['class_predicted'] = class_predicted
		session['predicted_proba_value'] = round(float(predicted_proba_value) * 100)

		return pred_details
		

	


