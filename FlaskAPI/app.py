import flask
from flask import Flask, jsonify, request
import json
<<<<<<< HEAD
import numpy as np
from data_input import data_in
import pickle


def load_models():
    model = pickle.load(open('./models/rnd_best.sav', 'rb'))
    return model
=======
>>>>>>> main

app = Flask(__name__)
@app.route('/predict', methods=['GET'])

def predict():
<<<<<<< HEAD
    # stub input features
    # parse input features from request
    request_json = request.get_json()
    x = request_json['input']
    x_in = np.array(x).reshape(1,-1)
    # load model
    model = load_models()
    prediction = np.exp(model.predict(x_in))
    prediction = prediction[0]
    response = json.dumps({'response': prediction})
    return response, 200
=======
 response = json.dumps({'response': 'yahhhh!'})
 return response, 200
>>>>>>> main

if __name__ == '__main__':
 application.run(debug=True)