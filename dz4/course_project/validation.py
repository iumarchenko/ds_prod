import pickle
import pandas as pd
from util.utils import plot_confusion_matrix, plot_PR_curve
from util.conf import COL_CNT
from matplotlib import pyplot as plt
import logging
LOGGER = logging.getLogger('app')
LOGGER.debug('Validate model')

with open('source/model.pkl', 'rb') as f:
    fitted_clf_2 = pickle.load(f)

with open('source/feature_importance.pkl', 'rb') as f:
    feature_importance = pickle.load(f)

dataset_train_balanced = pd.read_csv('dataset/dataset_train_balanced.csv', sep=';')
X_train_balanced = dataset_train_balanced.drop(['is_churned'], axis=1)
y_train_balanced = dataset_train_balanced['is_churned']

dataset_test = pd.read_csv('dataset/dataset_test_balanced.csv', sep=';')
X_test = dataset_test.drop(['is_churned'], axis=1)
y_test = dataset_test['is_churned']

X_train_FI = pd.DataFrame(X_train_balanced, columns=X_train_balanced.columns)[feature_importance[0][:COL_CNT]]
X_test_FI = pd.DataFrame(X_test, columns=X_test.columns)[feature_importance[0][:COL_CNT]]

predict_test = fitted_clf_2.predict(X_test_FI)
predict_test_probas = fitted_clf_2.predict_proba(X_test_FI)[:, 1]

plot_confusion_matrix(y_test.values, predict_test, classes=['churn', 'active'])
plt.show()

plot_PR_curve(y_test.values, predict_test, predict_test_probas)
plt.show()