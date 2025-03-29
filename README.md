# Titanic Survival Prediction

![Titanic Banner](https://img.shields.io/badge/Python-3.8+-blue.svg) ![License](https://img.shields.io/badge/License-MIT-green.svg)

A machine learning project to predict passenger survival on the Titanic using the Titanic dataset. This repository preprocesses the data, engineers new features, trains multiple models, and optimizes the best-performing model to achieve high accuracy.

---

## Project Overview

This project demonstrates a complete machine learning pipeline:
- **Data Preprocessing**: Handling missing values, encoding categorical variables, and normalizing numerical features.
- **Feature Engineering**: Creating meaningful features like `FamilySize` and `IsAlone`.
- **Model Training**: Comparing Logistic Regression, Random Forest, and XGBoost.
- **Hyperparameter Tuning**: Optimizing the Random Forest model with GridSearchCV.
- **Evaluation**: Reporting accuracy, precision, recall, and F1-score.

The final tuned model provides a robust prediction of survival based on passenger data.

---

## Dataset

The dataset used is `tested.csv` (Titanic dataset), which includes features like:
- `Pclass`: Passenger class
- `Sex`: Gender
- `Age`: Age of passenger
- `SibSp`: Number of siblings/spouses aboard
- `Parch`: Number of parents/children aboard
- `Fare`: Ticket fare
- `Cabin`: Cabin information
- `Embarked`: Port of embarkation

**Target**: `Survived` (0 = Did not survive, 1 = Survived)

---

## Requirements

- Python 3.8+
- Libraries:
  ```bash
  pip install pandas numpy scikit-learn xgboost
  ```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Project Structure

```
Titanic-Survival-Prediction/
├── data/
│   └── tested.csv         # Titanic dataset (not included, add your own)
├── main.py               # Main script to run the pipeline
├── README.md             # Project documentation
└── requirements.txt      # Dependencies
```

---

## How to Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/venkat-0706/Titanic-Survival-Prediction.git
   cd Titanic-Survival-Prediction
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Add dataset**:
   Place `tested.csv` in the `data/` folder (or update the file path in `main.py`).

4. **Run the script**:
   ```bash
   python main.py
   ```

The script will:
- Load and preprocess the data
- Engineer features
- Train and evaluate models
- Tune the best model and output the final accuracy

---

## Code Overview

### 1. Data Loading
Loads the dataset and displays initial insights (head, info, missing values).

### 2. Preprocessing
- Fills missing `Age` with median, `Embarked` with mode.
- Converts `Cabin` to binary (known/unknown).
- Encodes `Sex` (male: 0, female: 1) and `Embarked` (one-hot).
- Normalizes `Age` and `Fare`.

### 3. Feature Engineering
- `FamilySize`: Combines `SibSp` + `Parch` + 1.
- `IsAlone`: Flags passengers traveling alone.

### 4. Model Training
Trains and evaluates:
- Logistic Regression
- Random Forest
- XGBoost

Metrics: Accuracy, Precision, Recall, F1-Score.

### 5. Hyperparameter Tuning
Optimizes Random Forest with `GridSearchCV` using:
- `n_estimators`: [100, 200]
- `max_depth`: [10, 20, None]
- `min_samples_split`: [2, 5]

---

## Results

Sample output:
```
Logistic Regression:
Accuracy: 0.7857
Precision: 0.7500
Recall: 0.6667
F1-Score: 0.7059

Random Forest:
Accuracy: 0.8214
Precision: 0.8000
Recall: 0.7273
F1-Score: 0.7619

XGBoost:
Accuracy: 0.8036
Precision: 0.7778
Recall: 0.7000
F1-Score: 0.7368

Best Params: {'max_depth': 10, 'min_samples_split': 5, 'n_estimators': 200}
Best CV Score: 0.8345
Final Test Accuracy: 0.8393
```

*Note*: Results may vary based on your dataset split.

---

## Future Improvements
- Add visualization (e.g., confusion matrix, feature importance).
- Implement cross-validation for more robust evaluation.
- Experiment with additional features or models (e.g., SVM, Neural Networks).

---

## Contributing

Feel free to fork this repo, submit issues, or send pull requests. All contributions are welcome!

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Author**: Chandu Abbireddy  
**GitHub**: [github.com/Abbireddy Venkata Chandu](https://github.com/venkat-0706)  
**LinkedIn**: [linkedin.com/in/Abbireddy Venkata Chandu](https://linkedin.com/in/chandu0706)
