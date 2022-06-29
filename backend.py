from flask import Flask, redirect,render_template,request, url_for

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)
@app.route('/')
def front_page():
    return render_template('frontpage.html')

# @app.route("/<user>")
# def user(age):
#     return {{age}}

@app.route('/liver',methods=['GET','POST'])
def cancer_page():
    if request.method == 'GET':
        # age = request.form['age']
        # return redirect(url_for("user",usr=user))
        return render_template('cancer.html')
    else:
        # return render_template('cancer.html')
        age = request.form['age']
        sex = request.form['sex']
        total_bilirubin = request.form['chest']
        direct_bilirubin = request.form['trestbps']
        alkaline_phosphotase = request.form['chol']
        alamine_aminotransferase = request.form['fbs']
        aspartate_aminotransferase = request.form['restecg']
        total_proteins = request.form['thalach']
        albumin = request.form['exang']
        albumin_and_globulin_ratio = request.form['oldpeak']

        # return render_template('result.html',age=age)



        liver_dataset = pd.read_csv('indian_liver_patient.csv')
        liver_dataset['Gender'] = liver_dataset['Gender'].map({'Male': 1, 'Female': 2})
        liver_dataset.dropna(inplace=True)
        X = liver_dataset.drop(columns='Dataset', axis=1)
        Y = liver_dataset['Dataset']
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=101)
        model1 = RandomForestClassifier(n_estimators = 100)
        model1.fit(X_train, Y_train)
        input_data = (age,sex,total_bilirubin,direct_bilirubin,alkaline_phosphotase,alamine_aminotransferase,aspartate_aminotransferase,total_proteins,albumin,albumin_and_globulin_ratio)
        input_data_as_numpy_array= np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
        prediction = model1.predict(input_data_reshaped)
        senddata=""
        if (prediction[0]== 2):
            senddata='According to the given details person does not have Liver Disease'
        else:
            senddata='According to the given details chances of having Liver Disease are High, So Please Consult a Doctor'
        return render_template('result.html',resultvalue=senddata ,age=age, sex=sex , tb=total_bilirubin ,db=direct_bilirubin ,ak=alkaline_phosphotase,am=alamine_aminotransferase ,asam=aspartate_aminotransferase ,tpro=total_proteins ,alm=albumin , albratio=albumin_and_globulin_ratio)
        
        




if __name__ == '__main__':
    app.run()