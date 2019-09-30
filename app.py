from flask import Flask, render_template, request, redirect, session, jsonify
import config
from controllers import ModelsLoader, SessionManager, TrainStartManager, PredictStartManager
import pandas
import time
import pickle
import numpy as np
import json
import os.path
from flasgger import Swagger

app = Flask(__name__)
swagger = Swagger(app)

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


# -- TODO -- Create a blueprint for APIs. These APIs are for DEMO purpose only

@app.route('/api/trainstart', methods = ['POST'])
def api_trainstart():
    """Train from csv file
    Will train a model from data in csv file.
    ---
    parameters:
      - name: modelname_input
        type: string
        required: true
      - name: file
        type: binary
        required: true
      - name: targetname_input
        type: string
        required: true
      - name: description_textarea
        type: string
        required: true
      - name: train_time_input
        type: int
        required: true
    responses:
      200:
        description: status, message, accuracy, modelname
    """

    # - TODO - Apply async

    if request.method == 'POST':

        result = request.form
        modelname = result['modelname_input']
        targetname = result['targetname_input']
        model_desc = result['description_textarea']
        traning_time = int(result['train_time_input']) * 60

        df = pandas.read_csv(request.files.get('file'))

        status = TrainStartManager().start_training(targetname, df, modelname, model_desc, traning_time)
        
        if status['status'] == 'Error':
            return jsonify(status)

        return jsonify(status)

@app.route('/api/predict_csv/', methods = ['POST'])
def api_predict_csv():
    """Predict from csv file
    Will predict data from all rows in csv file.
    ---
    parameters:
      - name: modelselected
        type: string
        required: true
      - name: file
        type: binary
        required: true
    responses:
      200:
        description: Pandas.to_json() results of predictions
    """

    if request.method == 'POST':
        result = request.form
        modelselected = result['modelselected']

        df = pandas.read_csv(request.files.get('file'))
        for column in df.columns:
            if os.path.exists(config.MODELS_DIR + modelselected + '/{}.pkl'.format(column)):
                print('in')
                with open(config.MODELS_DIR + modelselected + '/{}.pkl'.format(column), 'rb') as pkl:
                    pkl = pickle.load(pkl)
                    df[column] = pkl.transform(df[column])

        with open(config.MODELS_DIR + modelselected + '/' + modelselected + '.pkl', 'rb') as model_pkl:
            model = pickle.load(model_pkl)
        with open(config.MODELS_DIR + modelselected + '/' + 'le_target.pkl', 'rb') as target_pkl:
            le_target = pickle.load(target_pkl)

        class_predicted = list(le_target.inverse_transform(model.predict(df)))
        predict_probabilities = model.predict_proba(df)

        maxInRows = np.amax(predict_probabilities, axis=1)

        dataset = pandas.DataFrame(class_predicted, columns =['Class'])
        probabilities = pandas.DataFrame(list(maxInRows), columns =['Probability'])
        
        status={}
        status['status'] = 'Success'
        status['results'] = dataset.join(probabilities).to_json()
        # print(pandas.read_json(dataset.join(probabilities).to_json()))
        return jsonify(status)

if __name__ == '__main__':
    app.run(debug=True)