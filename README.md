```markdown
# 🏦 SmartLender: AI-Powered Loan Approval System

SmartLender is a machine learning web application designed to streamline the loan approval process. By analyzing an applicant's financial and demographic data, the system instantly predicts loan eligibility using a highly optimized XGBoost classifier.

## 🌟 Highlights
- **Robust ML Pipeline:** Data preprocessing, Label Encoding, and Exploratory Data Analysis (EDA).
- **Hyperparameter Tuning:** Model optimized using `RandomizedSearchCV`.
- **Validated Performance:** Reliability verified using 5-Fold Cross-Validation.
- **Seamless Deployment:** End-to-end Flask web application with dynamic HTML/CSS front-end.

## 🛠️ Tech Stack
- **Language:** Python 3.12+
- **Machine Learning:** Scikit-Learn, XGBoost
- **Data Analysis:** Pandas, NumPy, Seaborn, Matplotlib
- **Web Framework:** Flask
- **Serialization:** Pickle

---

## 📁 Project Structure

```text
smartlender/
│
├── app.py                      # Flask backend (Routing & Prediction logic)
├── model_training.ipynb        # Jupyter notebook (EDA, CV, Tuning, Pickling)
├── smartlender_tuned_xgboost.pkl # Serialized XGBoost model
├── model_columns.pkl           # Serialized feature column names
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation (You are here)
│
└── templates/                  # HTML templates for Flask
    ├── home.html               # Landing page (Project details)
    ├── input.html              # Loan application form
    └── output.html             # Prediction results page
```

---

## 🚀 Setup & Installation

Follow these steps to get the project running on your local machine.

### Prerequisites
Make sure you have [Python](https://www.python.org/downloads/) installed on your system.

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/smartlender.git
cd smartlender
```

### 2. Create and Activate a Virtual Environment
*(Highly recommended to avoid dependency conflicts)*
```bash
# Create venv
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
All required Python packages are listed in the `requirements.txt` file. Install them using pip:
```bash
pip install -r requirements.txt
```

### 4. Run the Application
Start the Flask development server:
```bash
python app.py
```
The application will be available at: **http://127.0.0.1:5000/**

---

## 📊 Machine Learning Pipeline

1. **Data Preprocessing:** Handled missing values and converted categorical variables into numerical formats using Label Encoding.
2. **EDA:** Utilized Seaborn and Matplotlib to visualize relationships between features (e.g., `Gender` vs `ApplicantIncome`).
3. **Model Selection:** Compared Decision Tree, Random Forest, and XGBoost. XGBoost yielded the highest accuracy.
4. **Optimization:** Applied `RandomizedSearchCV` to find the optimal hyperparameters (`n_estimators`, `max_depth`, `learning_rate`, etc.).
5. **Validation:** Used 5-Fold Cross-Validation (`cross_val_score`) to ensure the model generalizes well to unseen data and avoids overfitting.
6. **Deployment:** Exported the tuned model and exact column structure using `pickle.dump()` for seamless integration with the Flask backend.

## 📝 Notes
- The ML model expects integer inputs. The Flask `app.py` script automatically converts the string data submitted from the HTML form into integers before passing it to XGBoost.
- The `model_columns.pkl` file ensures that the feature order passed from the web form exactly matches the order the model was trained on, preventing prediction errors.

## 📄 License
This project is licensed under the MIT License.
```

---

### 💡 Bonus: What to put inside your `requirements.txt`
If you haven't created your `requirements.txt` file yet, or want to make sure it's accurate, open your terminal (with your virtual environment activated) and run this command:

```bash
pip freeze > requirements.txt
```

It should automatically generate a file that looks something like this:
```text
Flask==3.0.0
numpy==1.26.2
pandas==2.1.4
scikit-learn==1.3.2
seaborn==0.13.0
matplotlib==3.8.2
xgboost==2.0.3
```