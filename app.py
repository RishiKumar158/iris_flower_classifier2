from flask import Flask, redirect, url_for, render_template, session
from wtforms import TextField, SubmitField
from flask_wtf import FlaskForm
import numpy as np
import joblib


def return_prediction(model, scaler, sample):
    s_len = sample['sepal_length']
    s_wid = sample['sepal_width']
    p_len = sample['petal_length']
    p_wid = sample['petal_width']

    flower = [[s_len, s_wid, p_len, p_wid]]

    flower = scaler.transform(flower)

    flower_classes = np.array(['setosa', 'versicolor', 'virginica'])

    class_ind = model.predict(flower)

    return flower_classes[class_ind][0]


app = Flask(__name__)
app.config['SECRET_KEY'] = 'myKey'


class FlowerForm(FlaskForm):
    sep_len = TextField('Sepal Length')
    sep_wid = TextField('Sepal Width')
    pet_len = TextField('Petal Length')
    pet_wid = TextField('Petal Width')
    submit = SubmitField('Analyze')


@app.route('/', methods=['GET', 'POST'])
def index():
    form = FlowerForm()

    if form.validate_on_submit():
        session['sep_len'] = form.sep_len.data
        session['sep_wid'] = form.sep_wid.data
        session['pet_len'] = form.pet_len.data
        session['pet_wid'] = form.pet_wid.data

        return redirect(url_for('prediction'))
    return render_template('home.html', form=form)


flower_model = joblib.load('iris_svm_model.joblib')
flower_scaler = joblib.load('iris_svm_scaler.pkl')


@app.route('/prediction')
def prediction():
    content = {'sepal_length': float(session['sep_len']), 'sepal_width': float(session['sep_wid']),
               'petal_length': float(session['pet_len']), 'petal_width': float(session['pet_wid'])}

    results = return_prediction(flower_model, flower_scaler, content)

    return render_template('prediction.html', results=results)


if __name__ == '__main__':
    app.run()
