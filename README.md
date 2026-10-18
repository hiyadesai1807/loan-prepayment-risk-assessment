# Loan Prepayment Risk Assessment (LendingClub)

Binary classification on LendingClub loan data with 18 engineered features (including loan-to-income ratio and credit-utilisation risk), benchmarking XGBoost against a Logistic Regression baseline.

## Version 1: first model (`01_first_version.ipynb`)
- Target: Fully Paid vs Charged Off, on 408,504 completed loans
- XGBoost: 0.735 ROC-AUC and 81% accuracy, vs 0.726 ROC-AUC for Logistic Regression
- On review, I found two problems:
  - "Fully Paid" includes loans repaid on schedule, so this target measures repayment vs default, not early prepayment
  - About 80% of loans were fully paid, so 81% accuracy was barely above always predicting the majority class

## Version 2: corrected prepayment model (`02_corrected_prepayment_model.ipynb`)
- Target: loans paid off at least 3 months before the end of their term vs loans paid on schedule
- Removed a timing bias: recent loans could only appear "fully paid" if paid early, so 418,085 unfinished loans were excluded, leaving 658,303 for modelling
- XGBoost: 0.618 ROC-AUC (62.5% accuracy) vs 0.604 for Logistic Regression; always predicting "prepaid" scores 61.3%
- Most important features: loan term, then the engineered credit-utilisation risk flag

## What I learned
Prepayment is much harder to predict than default from information available at loan issuance, because it depends on later events such as refinancing or income changes. Checking the target definition and the baseline changed the conclusion completely.

## Data
LendingClub accepted loans 2007–2018 (Kaggle: wordsforthewise/lending-club). Not included because of its size.

## Tools
Python, XGBoost, Scikit-learn, Pandas, Matplotlib, Seaborn
