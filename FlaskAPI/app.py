import flask
from flask import Flask, jsonify, request, render_template
import json
import numpy as np
import pickle


rnd_best = pickle.load(open('./models/rnd_best.sav', 'rb'))
dtree = pickle.load(open('./models/DecisionTree.sav', 'rb'))    

estimators = [rnd_best, dtree]

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():   
    #'rating','net_experience', 'jr', 'senior', 'bachelor', 'masters', 'posting_frequency'
     
    x_in = [int(x) for x in request.form.values()]
    x_in = np.array(x_in).reshape(1,-1)

    ans=[]
    for reg in estimators:
        pred = reg.predict(x_in)
        ans.append(pred[0])
    
    prediction = round(np.exp(sum(ans)/len(ans)), 2)
    
    return render_template('index.html', prediction_text='Your predicted salary is {}'.format(prediction))


if __name__ == '__main__':
 app.run(debug=True)