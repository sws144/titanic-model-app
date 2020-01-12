# flask app file

import pandas as pd
from flask import Flask, jsonify, request, render_template
import pickle

#load model
model = pickle.load(open('model.pkl','rb'))

# app
app = Flask(__name__, template_folder='templates')

# routes
@app.route('/',methods=['GET','POST'])
def main():
    if request.method == 'GET':
        return(render_template('main.html'))
    if request.method == 'POST':
        pclass = request.form['Pclass']
        age = request.form['Age']
        sibSp = request.form['SibSp']
        fare = request.form['Fare']        
        
        input_variables = pd.DataFrame([[pclass, age, sibSp, fare]],
                columns=['Pclass', 'Age', 'SibSp','Fare'],
                dtype=int)        
        
        prediction = model.predict(input_variables)[0]        
        
        return render_template('main.html',
                original_input={'Pclass': pclass,
                                'Age':age,
                                'Sibsp':sibSp,
                                'Fare': fare},
                result=prediction
        )

# def predict():
#     # get data
#     data = request.get_json(force = True)
    
#     # convert to dataframe
#     data.update((x,[y]) for x , y in data.items())  
#     data_df = pd.DataFrame.from_dict(data)
    
#     # predictions
#     result = model.predict(data_df)
    
#     # send back to browser
#     output  = {'results': int(result[0])}
    
#     # return data
#     return jsonify(result=output)



if __name__ == '__main__':
    app.run()