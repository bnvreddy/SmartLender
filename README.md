# 🏦 SmartLender: AI-Powered Loan Approval System

SmartLender is a machine learning web application designed to streamline the loan approval process. By analyzing an applicant's financial and demographic data, the system instantly predicts loan eligibility using a highly optimized XGBoost classifier.

## 🌟 Highlights
- **Robust ML Pipeline:** Data preprocessing, Label Encoding, and Exploratory Data Analysis (EDA).
- **Hyperparameter Tuning:** Model optimized using `RandomizedSearchCV`.
- **Validated Performance:** Reliability verified using 5-Fold Cross-Validation (Achieved 81.11% CV Accuracy).
- **Modular Architecture:** Clean separation of datasets, training scripts, production-ready Flask backend, and project documentation.
- **Seamless Deployment:** End-to-end Flask web application with dynamic HTML/CSS front-end.

## 🛠️ Tech Stack
- **Language:** Python 3.12+
- **Machine Learning:** Scikit-Learn, XGBoost
- **Data Analysis:** Pandas, NumPy, Seaborn, Matplotlib
- **Web Framework:** Flask
- **Serialization:** Pickle

---

## 📁 Project Structure

The project is strictly divided into core directories for seamless workflow management and academic documentation:

```text
smartlender/
│
├── Dataset/                        # Raw data storage
│   └── loan_prediction.csv     
│
├── Training/                       # Data science and model building
│   └── model_training.ipynb        # EDA, RandomizedSearchCV, and model pickling
│
├── Flask/                          # Web application and deployment
│   ├── app.py                      # Flask backend (Routing & Prediction logic)
│   ├── smartlender_tuned_xgboost.pkl # Serialized XGBoost model
│   ├── model_columns.pkl           # Serialized feature column names
│   └── templates/                  # HTML templates for Flask
│       ├── home.html               # Landing page (Project details)
│       ├── predict.html            # Loan application form
│       └── output.html             # Prediction results page
│
├── Project_Documentation/          # Phase-wise project reports (PDFs)
│   ├── 1.Brainstorming_&_Ideation/
│   ├── 2.Requirement_Analysis/
│   ├── 3.Project_Design_Phase/
│   ├── 4.Project_Planning_Phase/
│   ├── 5.Project_Development_Phase/
│   ├── 6.Project_Testing/
│   ├── 7.Project_Documentation/
│   └── 8.Project_Demonstration/
│
├── requirements.txt                # Python dependencies for the web app
└── README.md                       # Project documentation
```

---

## 🚀 Setup & Installation

Follow these steps to get the web application running on your local machine.

### Prerequisites
Make sure you have [Python](https://www.python.org/downloads/) installed on your system.

### 1. Clone the Repository
```bash
git clone https://github.com/bnvreddy/SmartLender.git
cd SmartLender
```

### 2. Create and Activate a Virtual Environment
Set up a virtual environment to avoid dependency conflicts.
```bash
# Create venv
python -m venv .venv

# Activate on Windows
.venv\Scripts\activate

# Activate on macOS/Linux
source .venv/bin/activate
```

### 3. Install Dependencies
All required Python packages for the web app are listed inside the `Flask/` directory. Install them using pip:
```bash
pip install -r requirements.txt
```

### 4. Run the Application
Navigate into the Flask directory and start the development server:
```bash
cd Flask
python app.py
```
The application will be available at: **http://127.0.0.1:5000/**

---

## 📊 Machine Learning Workflow

*(Performed inside `Training/model_training.ipynb`)*

1. **Data Loading:** Ingested `loan_prediction.csv` from the `Dataset/` folder (614 rows, 13 features).
2. **Data Preprocessing:** Handled missing values and converted all categorical variables into `int64` formats using Label Encoding to ensure strict compatibility with XGBoost.
3. **EDA:** Utilized Seaborn and Matplotlib to visualize relationships between features (e.g., `Gender` vs `ApplicantIncome`, distribution plots).
4. **Model Selection:** Compared Decision Tree, Random Forest, and XGBoost. XGBoost yielded the highest baseline accuracy.
5. **Optimization:** Applied `RandomizedSearchCV` (50 iterations) to find the optimal hyperparameters (`subsample: 0.7`, `n_estimators: 300`, `max_depth: 5`, `learning_rate: 0.01`, `colsample_bytree: 1.0`).
6. **Validation & Metrics:** 
   - **Best 5-Fold CV Accuracy:** 81.11%
   - **Final Training Accuracy:** 100%
   - **Final Testing Accuracy:** 79.67%
7. **Export:** Exported the tuned model and exact column structure using `pickle.dump()`, saving the files directly into the `Flask/` directory for immediate deployment.

## 📝 Notes
- The ML model strictly expects `int64` inputs. The Flask `app.py` script automatically converts the string data submitted from the HTML form into integers before passing it to XGBoost.
- The `model_columns.pkl` file ensures that the feature order passed from the web form exactly matches the order the model was trained on, preventing silent prediction errors.