# Task 2: SVM and kNN 

## Optimal configuration
The data was splitted on train and test sets in proportion 80:20

| Classifier  | Configuration                                                   |
|-------------|-----------------------------------------------------------------|
| SVM         | {'C': 10, 'degree': 2, 'gamma': 1, 'kernel': 'rbf'}             |
| kNN         | {'metric': 'manhattan', 'n_neighbors': 1, 'weights': 'uniform'} |

## Average accuracy score
Average accuracy score for 100 launches (for train and test sets)

| Classifier  | Train set | Test set |
|-------------|-----------|----------|
| SVM         | 0.8366    | 0.8383   |
| kNN         | 1.0       | 1.0      |
