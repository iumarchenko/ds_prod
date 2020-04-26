import pickle
import pandas as pd
from util.utils import prepare_dataset
from util.conf import COL, COL_CNT
import logging
LOGGER = logging.getLogger('app')

LOGGER.debug('Create pridiction')
test = pd.read_csv('dataset/dataset_raw_test.csv', sep=';')

with open('source/imputer.pkl', 'rb') as f:
    imp = pickle.load(f)
prepare_dataset(dataset=test,imp=imp,COL=COL, dataset_type='test')


with open('source/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
X = test.drop(['user_id'], axis=1)
X_test = scaler.fit_transform(X)
X_test = pd.DataFrame(X_test, columns=X.columns)

with open('source/model.pkl', 'rb') as f:
    fitted_clf_2 = pickle.load(f)

with open('source/feature_importance.pkl', 'rb') as f:
    feature_importance = pickle.load(f)

X_test_FI = pd.DataFrame(X_test, columns=X_test.columns)[feature_importance[0][:COL_CNT]]

predict_test = fitted_clf_2.predict(X_test_FI)
predict_proba_test = fitted_clf_2.predict_proba(X_test_FI)

result = pd.concat([test['user_id'], pd.Series(predict_test)], axis=1)
result = result.rename(columns={0: 'is_churned'})
print(result['is_churned'].value_counts())

result.to_csv('source/IMarchenko_predictions.csv',index=None)

LOGGER.debug('Prediction save to source/IMarchenko_predictions.csv')

