from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Route for home page
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
                passenger_count=int(request.form.get('passenger_count')),
                distance=float(request.form.get('distance')),
                is_rush_hour=int(request.form.get('is_rush_hour')),
                is_a_group_trip=int(request.form.get('is_a_group_trip')),
                fare_per_km=float(request.form.get('fare_per_km')),
                fare_per_passenger=float(request.form.get('fare_per_passenger')),
                passenger_count_log=float(request.form.get('passenger_count_log'))
            )
            
            pred_df = data.get_data_as_data_frame()
            print("Input DataFrame:")
            print(pred_df)

            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)

            return render_template('home.html', results=f"${results[0]:.2f}")
        
        except Exception as e:
            return render_template('home.html', results=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)