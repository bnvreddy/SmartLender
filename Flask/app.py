from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

with open('smartlender_tuned_xgboost.pkl', 'rb') as file:
    model = pickle.load(file)

with open('model_columns.pkl', 'rb') as file:
    model_columns = pickle.load(file)


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/apply')
def apply():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        form_data = request.form.to_dict()
        input_data = {key: int(value) for key, value in form_data.items()}
        input_df = pd.DataFrame([input_data])
        input_df = input_df[model_columns]
        
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)[0][1] * 100
        
        if prediction[0] == 1:
            result_text = "APPROVED"
            result_color = "green"
        else:
            result_text = "REJECTED"
            result_color = "red"

        return render_template('output.html', 
                               prediction=result_text, 
                               probability=round(probability, 2),
                               color=result_color)

    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)