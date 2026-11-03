# Loan Default Predictor

Predicts the chance that a borrower defaults on a loan. Built in Java with Weka (training) and Spring Boot (web API + page).

## Dataset
Kaggle "Credit Risk Dataset" (target: `loan_status`, 0 = repaid, 1 = defaulted). Defaults are a minority class, so accuracy is not used; models are compared with ROC-AUC, precision, recall and F1 on a held-out 20% test set.

## Results
| Model | AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.872 | 0.758 | 0.554 | 0.641 |
| Random Forest | 0.928 | 0.963 | 0.735 | 0.834 |

Evaluated on a held-out 20% test set (6,516 loans). Random Forest scored best and is used by the web app.

## Run it
1. Put `loan.csv` (from Kaggle) in `data/`.
2. Run `Train.java` (trains both models, saves the better one to `src/main/resources/loan.model`).
3. Run `LoanApplication.java` and open http://localhost:8080

## Next improvements
Handle class imbalance (cost-sensitive learning), tune hyperparameters on a validation split, add feature-importance analysis.
