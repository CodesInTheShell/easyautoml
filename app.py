from flask import Flask, render_template, request, redirect, session
import config
from controllers import ModelsLoader, SessionManager, TrainStartManager, PredictStartManager
import pandas
import time
import pickle
import numpy as np
import json


app = Flask(__name__)

app.secret_key = config.ML_APP_SECRET_KEY

@app.route('/')
def index():
	# session.clear()
	session.pop('modelselected', None)
	session.pop('predicted_value', None)
	models = ModelsLoader().load_all_models()
	session['models'] = models
	return render_template("index.html")

@app.route('/train')
def train():
	return render_template("train.html")

@app.route('/predict')
def predict():

	return render_template("predict.html")

@app.route('/trainstart', methods = ['POST'])
def trainstart():

	# - TODO - Apply async

	if request.method == 'POST':

		result = request.form
		modelname = result['modelname_input']
		targetname = result['targetname_input']
		model_desc = result['description_textarea']
		traning_time = int(result['train_time_input']) * 60

		f = request.files['file']

		df = pandas.read_csv(request.files.get('file'))

		status = TrainStartManager().start_training(targetname, df, modelname, model_desc, traning_time)
		
		if status['status'] == 'Error':
			return status['message']
		
		return redirect('/')

@app.route('/predicstart', methods = ['POST'])
def predicstart():

	pred_details = PredictStartManager.start_predict()

	return redirect("predict")


@app.route('/modelselect', methods = ['POST'])
def modelselect():
	session.pop('modelselected', None)
	# session.pop('predicted_value', None)
	session.pop('class_predicted', None)
	session.pop('predicted_proba_value', None)

	modelselected = request.form['modelselected']
	session['modelselected'] = modelselected

	modeldir = config.MODELS_DIR + modelselected
	print('modeldir: ', modeldir)

	model_info_json = modeldir + '/model_info.json'
	print('model_info_json: ', model_info_json)

	with open(model_info_json) as json_file:
		data = json.load(json_file)
		print(round(float(data['accuracy_score']) * 100))
		print(data['description'])

	SessionManager().add_to_session(session, 'model_acc', round(float(data['accuracy_score']) * 100))
	SessionManager().add_to_session(session, 'model_desc', data['description'])

	columns_txt = model_info_json = modeldir + '/columns.txt'
	columns = []
	with open(columns_txt) as txt:
		columns = [c.strip() for c in txt]
	SessionManager().add_to_session(session, 'model_cols', columns)

	return redirect('/predict')

if __name__ == '__main__':
	app.run(debug=True)