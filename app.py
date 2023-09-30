from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from flask_cors import CORS 

app = Flask(__name__)
CORS(app)

# Load and preprocess the data
df = pd.read_csv('city_day.csv', na_values='=')
data2 = df.copy()
data2 = data2.fillna(data2.mean())

# Mapping for City and AQI_Bucket columns
city_mapping = {city: i for i, city in enumerate(data2['City'].unique())}
data2['City'] = data2['City'].map(city_mapping)

aqi_bucket_mapping = {bucket: i for i, bucket in enumerate(data2['AQI_Bucket'].unique())}
data2['AQI_Bucket'] = data2['AQI_Bucket'].map(aqi_bucket_mapping)
data2['AQI_Bucket'] = data2['AQI_Bucket'].fillna(data2['AQI_Bucket'].mean())

data2 = data2.drop(['Date', 'AQI_Bucket'], axis=1)

# IQR-based outlier removal
Q1 = np.percentile(data2['AQI'], 25, interpolation="midpoint")
Q3 = np.percentile(data2['AQI'], 75, interpolation='midpoint')
IQR = Q3 - Q1

upper = np.where(data2['AQI'] >= (Q3 + 1.5 * IQR))
lower = np.where(data2['AQI'] <= (Q1 - 1.5 * IQR))

data2.drop(upper[0], inplace=True)
data2.drop(lower[0], inplace=True)

# Feature selection and data splitting
features = data2[['City', 'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3',
       'Benzene', 'Toluene', 'Xylene']]
labels = data2['AQI']

Xtrain, Xtest, Ytrain, Ytest = train_test_split(features, labels, test_size=0.2, random_state=2)

# Train the RandomForestRegressor model
regr = RandomForestRegressor(max_depth=300, random_state=10)
regr.fit(Xtrain, Ytrain)
@app.route('/predict_aqi', methods=['POST'])
def predict_aqi():
    try:
        # Get input data as JSON
        input_data = request.get_json()
        print("Received input data:", input_data)

        # Extract input features
        input_features = input_data['input']
        input_features[0] = 1
        print("Input features:", input_features)

        # Make a prediction using the trained model
        prediction = regr.predict([input_features])
        print("Prediction:", prediction)

        # Calculate R-squared score
        r2 = r2_score(Ytest, regr.predict(Xtest))
        print("R-squared score:", r2)

        # Prepare the response JSON
        response = {
            'prediction': prediction[0],
            'r2_score': r2
        }
        print("Response:", response)
        return jsonify(response)
    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
