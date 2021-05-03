import flask
from flask import Flask, jsonify, request, render_template
import json
import numpy as np
import pickle


best_model = pickle.load(open('./models/best_model.sav', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():   
    #'rating','net_experience', 'jr', 'senior', 'bachelor', 'masters', 'posting_frequency'
     
    x_in = [float(x) for x in request.form.values()]
    x_in = np.array(x_in).reshape(1,-1)

    pred = best_model.predict(x_in)
    
    prediction = np.round(np.exp(pred), 2)
    
    return render_template('index.html', prediction_text='Your predicted annual salary is {}'.format(prediction))


if __name__ == '__main__':
 app.run(debug=True)