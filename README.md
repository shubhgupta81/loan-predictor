# Loan Default Predictor

Predicts the chance that a borrower defaults on a loan. Built in Java with Weka (training) and Spring Boot (web API + page).

## Dataset
Kaggle "Credit Risk Dataset" (target: `loan_status`, 0 = repaid, 1 = defaulted). Defaults are a minority class, so accuracy is not used; models are compared with ROC-AUC, precision, recall and F1 on a held-out 20% test set.

## Results
| Model | AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | _fill in_ | | | |
| Random Forest | _fill in_ | | | |

## Run it
1. Put `data/loan.csv` in `data/`.
2. Run `Train.java` (trains both models, saves the better one to `src/main/resources/loan.model`).
3. Run `LoanApplication.java` and open http://localhost:8080

## Next improvements
Handle class imbalance (cost-sensitive learning), tune hyperparameters on a validation split, add feature-importance analysis.
