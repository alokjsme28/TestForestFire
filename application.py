import pickle
import pandas as pd
from flask import Flask,request, jsonify, render_template

application = Flask(__name__)
app = application

# Importing the pickle models
ridge = pickle.load(open('models/ridge.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_data', methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        Temperature = request.form.get('Temperature')
        Rh = request.form.get('Rh')
        Ws = request.form.get('Ws')
        Rain = request.form.get('Rain')
        FFMC = request.form.get('FFMC')
        DMC = request.form.get('DMC')
        ISI = request.form.get('ISI')
        Classes = request.form.get('Classes')
        Region = request.form.get('Region')

        data = pd.DataFrame([[Temperature,Rh,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        data_scaled = scaler.transform(data)
        result = ridge.predict(data_scaled)
        return render_template('home.html',results = result[0])

    else:
        return render_template('home.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)