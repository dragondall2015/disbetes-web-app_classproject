import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import googleapiclient.discovery
import os
from flask import Flask, render_template
from dotenv import load_dotenv

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hard to guess string'

from flask_bootstrap import Bootstrap
bootstrap5 = Bootstrap(app)

from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
from tensorflow import keras

# 사용자 입력 폼 클래스 정의
class LabForm(FlaskForm):
    preg    = StringField('# Pregnancies', validators=[DataRequired()])
    glucose = StringField('Glucose', validators=[DataRequired()])
    blood   = StringField('Blood pressure', validators=[DataRequired()])
    skin    = StringField('Skin thickness', validators=[DataRequired()])
    insulin = StringField('Insulin', validators=[DataRequired()])
    bmi     = StringField('BMI', validators=[DataRequired()])
    dpf     = StringField('DPF Score', validators=[DataRequired()])
    age     = StringField('Age', validators=[DataRequired()])
    submit  = SubmitField('Submit')

# 기본 라우팅
@app.route('/', methods=['GET', 'POST'])
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['GET', 'POST'])
def lab():
    form = LabForm()
    if form.validate_on_submit():
        # 사용자의 입력 데이터를 배열로 구성
        X_test = np.array([[
            float(form.preg.data),
            float(form.glucose.data),
            float(form.blood.data),
            float(form.skin.data),
            float(form.insulin.data),
            float(form.bmi.data),
            float(form.dpf.data),
            float(form.age.data)
        ]])

        print(X_test.shape)
        print(X_test)

        # 전체 데이터 로딩 (정규화 기준)
        data = pd.read_csv('./diabetes.csv', sep=',')
        X = data.values[:, 0:8]
        y = data.values[:, 8]

        # MinMaxScaler로 정규화
        scaler = MinMaxScaler()
        scaler.fit(X)
        X_test = scaler.transform(X_test)

        # 모델 불러오기
        model = keras.models.load_model('pima_model.keras')

        # 예측 수행
        prediction = model.predict(X_test)
        res = prediction[0][0]
        res = np.round(res, 2)
        res = float(np.round(res * 100))  # 확률(%)로 변환

        return render_template('result.html', res=res)

    return render_template('prediction.html', form=form)

if __name__ == '__main__':
    app.run()
