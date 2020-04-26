from util.utils import prepare_dataset
from util.conf import COL
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import logging
import log.config_log
import warnings
warnings.filterwarnings("ignore")

LOGGER = logging.getLogger('app')
LOGGER.debug('Prepare dataset')

dataset_train = pd.read_csv('dataset/dataset_raw_train.csv', sep=';')
imp = IterativeImputer(max_iter=10, verbose=0)
prepare_dataset(dataset=dataset_train, imp=imp, COL=COL, dataset_type='train')


