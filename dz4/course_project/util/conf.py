import logging

# Следует из исходных данных
CHURNED_START_DATE = '2019-09-01'
CHURNED_END_DATE = '2019-10-01'

INTER_1 = (1, 7)
INTER_2 = (8, 14)
INTER_3 = (15, 21)
INTER_4 = (22, 28)
INTER_LIST = [INTER_1, INTER_2, INTER_3, INTER_4]


COL = [
    'age',
    'gender',
    'days_between_fl_df',
    'days_between_reg_fl',
    'level',
    'donate_total',
    'has_return_date',
    'has_phone_number'
]

COL_CNT = 30

LOGGING_LEVEL = logging.DEBUG