from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Load the pickled model and columns ONCE when the app starts
with open('smartlender_tuned_xgboost.pkl', 'rb') as file:
    model = pickle.load(file)

with open('model_columns.pkl', 'rb') as file:
    model_columns = pickle.load(file)

@app.route('/')
def home():
    # This renders the input form
    return render_template('input.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Get data from the HTML form
        form_data = request.form.to_dict()
        
        # 2. Convert all values to integers (HTML forms send strings!)
        # If you don't do this, XGBoost will crash
        input_data = {key: int(value) for key, value in form_data.items()}
        
        # 3. Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # 4. FORCE the column order to match exactly what the model was trained on
        input_df = input_df[model_columns]
        
        # 5. Make the prediction
        prediction = model.predict(input_df)
        
        # 6. Get the probability (How confident is the model?)
        # predict_proba returns [prob_of_0, prob_of_1]
        probability = model.predict_proba(input_df)[0][1] * 100
        
        # 7. Format the output text based on the dataset's encoding (Usually 1 = Approved, 0 = Rejected)
        if prediction[0] == 1:
            result_text = "APPROVED"
            result_color = "green"
        else:
            result_text = "REJECTED"
            result_color = "red"

        # 8. Send the results to the output page
        return render_template('output.html', 
                               prediction=result_text, 
                               probability=round(probability, 2),
                               color=result_color)

    except Exception as e:
        # If there's an error, show it on the page (helpful for debugging)
        return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)