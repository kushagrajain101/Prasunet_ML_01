from flask import Flask, request, render_template
from joblib import load
import pandas as pd
import traceback

app = Flask(__name__)

# Load the saved model
try:
    model = load('house_price_model_simple.joblib')
except Exception as e:
    print(f"Error loading the model: {e}")
    traceback.print_exc()

@app.route('/')
def home():
    try:
        return render_template('index.html')
    except Exception as e:
        print(f"Error rendering home page: {e}")
        traceback.print_exc()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input values from the form
        GrLivArea = float(request.form['GrLivArea'])
        BedroomAbvGr = int(request.form['BedroomAbvGr'])
        FullBath = int(request.form['FullBath'])
        HalfBath = int(request.form['HalfBath'])

        # Create new feature for total bathrooms
        TotalBath = FullBath + 0.5 * HalfBath

        # Create a DataFrame with the input values
        input_data = pd.DataFrame({
            'GrLivArea': [GrLivArea],
            'BedroomAbvGr': [BedroomAbvGr],
            'TotalBath': [TotalBath]
        })

        # Make prediction
        prediction = model.predict(input_data)

        return render_template('index.html', prediction_text=f'Predicted Sale Price: ${prediction[0]:,.2f}')
    except Exception as e:
        print(f"Error in prediction: {e}")
        traceback.print_exc()
        return render_template('index.html', prediction_text="An error occurred during prediction. Please check the inputs and try again.")
if __name__ == "__main__":
    try:
        app.run(host='0.0.0.0', port=8080, debug=True, use_reloader=False)
    except Exception as e:
        print(f"Error starting the Flask app: {e}")
        traceback.print_exc()
