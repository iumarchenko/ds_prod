import pickle
import pandas as pd
from util.utils import xgb_fit_predict, plot_importance
from util.conf import COL_CNT
import logging
LOGGER = logging.getLogger('app')

LOGGER.debug('Create model')
dataset_train_balanced = pd.read_csv('dataset/dataset_train_balanced.csv', sep=';')
X_train_balanced = dataset_train_balanced.drop(['is_churned'], axis=1)
y_train_balanced = dataset_train_balanced['is_churned']

dataset_test = pd.read_csv('dataset/dataset_test_balanced.csv', sep=';')
X_test = dataset_test.drop(['is_churned'], axis=1)
y_test = dataset_test['is_churned']

fitted_clf = xgb_fit_predict(X_train_balanced, y_train_balanced, X_test, y_test)

feature_importance = plot_importance(fitted_clf.feature_importances_, X_train_balanced.columns, 'Features Importance')

with open('source/feature_importance.pkl', 'wb') as file:
    pickle.dump(feature_importance, file)

X_train_FI = pd.DataFrame(X_train_balanced, columns=X_train_balanced.columns)[feature_importance[0][:COL_CNT]]
X_test_FI = pd.DataFrame(X_test, columns=X_test.columns)[feature_importance[0][:COL_CNT]]

fitted_clf_2 = xgb_fit_predict(X_train_FI, y_train_balanced, X_test_FI, y_test)

print('Признаков было:', X_train_balanced.shape[1])
print('Признаков стало:', X_train_FI.shape[1])

with open('source/model.pkl', 'wb') as file:
    pickle.dump(fitted_clf_2, file)

