import pickle
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
import logging
warnings.filterwarnings("ignore")

LOGGER = logging.getLogger('app')
LOGGER.debug('Scale dataset')

dataset_train = pd.read_csv('dataset/dataset_train.csv', sep=';')

# балансировка
X = dataset_train.drop(['user_id', 'is_churned'], axis=1)
y = dataset_train['is_churned']

scaler = MinMaxScaler()
X_mm = scaler.fit_transform(X)
with open('source/scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

X_train, X_test, y_train, y_test = train_test_split(X_mm,
                                                    y,
                                                    test_size=0.3,
                                                    shuffle=True,
                                                    stratify=y,
                                                    random_state=100)

# Снизим дизбаланс классов
smote_on_1 = int(X_train.shape[0] * 3 / 10)
X_train_balanced, y_train_balanced = SMOTE(random_state=42, sampling_strategy={1: smote_on_1}).fit_sample(X_train, y_train)

X_train_balanced = pd.DataFrame(X_train_balanced, columns=X.columns)
X_train_balanced['is_churned'] = y_train_balanced.values
X_train_balanced.to_csv('dataset/dataset_train_balanced.csv', sep=';', index=False)

X_test = pd.DataFrame(X_test, columns=X.columns)
X_test['is_churned'] = y_test.values
X_test.to_csv('dataset/dataset_test_balanced.csv', sep=';', index=False)