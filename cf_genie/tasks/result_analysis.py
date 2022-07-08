import cf_genie.logger as logger
from cf_genie.models import SUPERVISED_MODELS
from cf_genie.models.model_analyisis import \
    get_acc_pandas_df_for_model_all_classes

logger.setup_applevel_logger(
    is_debug=False, file_name=__file__, simple_logs=True)


log = logger.get_logger(__name__)


def main():
    for model_class in SUPERVISED_MODELS:
        log.info('Analysing results of model %s', model_class.__name__)
        df_f1_micro = get_acc_pandas_df_for_model_all_classes(model_class, ['f1_micro'])
        df_hamming = get_acc_pandas_df_for_model_all_classes(model_class, ['hamming_score'])
        print(df_f1_micro)
        print(df_hamming)
    pass


if __name__ == '__main__':
    main()
