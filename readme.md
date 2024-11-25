# Heart Disease Analysis App

This Streamlit application provides an interactive interface for analyzing heart disease data using machine learning techniques. The app includes data visualization, model training, and performance metrics.

## Features

- Interactive data exploration
- Random Forest model training
- Feature importance analysis
- Model performance visualization
- Confusion matrix and ROC curve analysis

## Setup Instructions

1. Clone this repository:
```bash
git clone [your-repository-url]
cd heart-disease-analysis
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Download the dataset:
- Create a `data` folder in the project directory
- Download the `cardio_train.csv` file from [this link](https://raw.githubusercontent.com/datascintist-abusufian/Post-hoc-explanation-cardio-phenotype-interpretability/main/cardio_train.csv)
- Place the downloaded file in the `data` folder

5. Run the application:
```bash
streamlit run app.py
```

## Data Description

The dataset includes the following features:
- Age (in days)
- Gender (0: Female, 1: Male)
- Height (cm)
- Weight (kg)
- Systolic blood pressure
- Diastolic blood pressure
- Cholesterol levels
- Glucose levels
- Smoking status
- Alcohol consumption
- Physical activity
- Presence of cardiovascular disease (target variable)

## Project Structure

```
heart-disease-analysis/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md          # Project documentation
└── data/
    └── cardio_train.csv  # Dataset file
```

## Requirements

- Python 3.8+
- See requirements.txt for Python package dependencies
