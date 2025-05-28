from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from src.logger import logging
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('race_ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=float(request.form.get('reading_score')),
                writing_score=float(request.form.get('writing_score'))
            )
            logging.info("Created CustomData object")
            
            pred_df = data.get_data_as_data_frame()
            logging.info(f"Created DataFrame for prediction: {pred_df}")

            predict_pipeline = PredictPipeline()
            logging.info("Making prediction")
            
            results = predict_pipeline.predict(pred_df)
            logging.info(f"Prediction complete. Result: {results[0]}")
            
            return render_template('home.html', results=results[0])
            
        except Exception as e:
            logging.error(f"Error in prediction: {str(e)}")
            return render_template('home.html', error=str(e))

if __name__ == "__main__":
    app.run(host="0.0.0.0")

