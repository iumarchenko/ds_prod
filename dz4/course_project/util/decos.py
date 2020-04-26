import sys
import logging
import log.config_log
import traceback
import inspect

LOGGER = logging.getLogger('app')

def log(func_to_log):
    def log_saver(*args, **kwargs):
        ret = func_to_log(*args, **kwargs)
        LOGGER.debug(f'Была вызвана функция {func_to_log.__name__}. '
                     f'Вызов из модуля {func_to_log.__module__}. ')
        return ret
    return log_saver
