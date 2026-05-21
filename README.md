# Titanic Survival Prediction — ML Algorithms Comparison

Started this as a simple EDA project but the plan was always to go further — compare multiple classification algorithms on the same data and see how they actually stack up against each other.

---

## What's in the notebook

**1. EDA**
Plotted survival rates across gender, passenger class, and age groups — all in one figure using subplots. Also did a correlation heatmap after mapping Sex to numbers. The patterns are pretty much what you'd expect (women and first class passengers survived more), but it's good to see it visually.

**2. Preprocessing**
- Dropped `Cabin` (too many nulls), `Name`, `PassengerId`, `Ticket` (not useful)
- Filled missing `Age` values with median — done after the train/test split to avoid leakage
- Created `AgeGroup` bins: child / teen / adult / senior
- One-hot encoded `Embarked`, mapped `Sex` to 0/1

**3. Models trained**
Ran all 5 in a loop instead of manually fitting each one:
- Logistic Regression
- KNN
- Decision Tree
- Random Forest
- SVM

**4. Evaluation**
Compared all models on accuracy, precision, recall, and F1. Results in a sorted DataFrame + horizontal bar chart.

**5. Visualizations**
- Confusion matrices for all 5 models in a grid
- Decision tree plotted with `max_depth=3` (full tree is unreadable)
- Random Forest feature importances — `Fare` and `Age` ended up being the most important

---

## How to run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Open `titanic_analysis.ipynb` and run all cells. Dataset is loaded directly from GitHub so no local CSV needed.

---

## Results

Random Forest and SVM come out on top at around 82-83% accuracy. Logistic Regression is surprisingly close behind. KNN is the weakest of the bunch on this dataset.