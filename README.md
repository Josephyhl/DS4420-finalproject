# Breast Cancer Classification: Neural Network vs. Bayesian Logistic Regression

**Course:** DS 4420 – Advanced Machine Learning
**Authors:** Zichen Mi, Hanlin Ye

## What this project is about

We wanted to see how well two different models can tell malignant tumors from benign ones, and whether a simple linear Bayesian model can keep up with a neural network.

- **MLP:** a small neural network (30 inputs, 16 hidden ReLU neurons, 1 sigmoid output) trained with backpropagation.
- **Bayesian logistic regression (MAP):** logistic regression with a Gaussian prior on the weights (variance 10), trained with gradient descent.

We wrote both models from scratch in NumPy. scikit-learn only shows up for loading the data, splitting it, standardizing features, and computing metrics.

## Data

We used the Wisconsin Diagnostic Breast Cancer dataset (569 samples, 30 features, 212 malignant and 357 benign), loaded with `sklearn.datasets.load_breast_cancer`.

- 80/20 stratified split with `random_state=42` (455 train, 114 test)
- Features standardized using only the training set's mean and std, so nothing leaks from the test set

## Results (114 test samples)

| Metric | MLP | Bayesian MAP |
| --- | --- | --- |
| Accuracy | 96.49% | 97.37% |
| Precision | 93.18% | 95.35% |
| Recall | 97.62% | 97.62% |
| F1 Score | 95.35% | 96.47% |

Precision, recall, and F1 treat malignant as the positive class. Both models caught 41 of the 42 malignant cases. Bayesian MAP is ahead by just one test sample, so we can't say one method is really better from a single split.

## What's in this repo

- `DS4420_Final_Project.ipynb`: all the code (data prep, both models, evaluation, plots)
- `model_comparison.png`: bar chart of the two models' test metrics
- `README.md`: this file

## How to run it

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
jupyter notebook DS4420_Final_Project.ipynb
```

Just run the cells from top to bottom. Seeds are fixed, so you should get the same numbers.

## Reference

Wolberg, W. H., Street, W. N., & Mangasarian, O. L. (1995). *Breast Cancer Wisconsin (Diagnostic)* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B