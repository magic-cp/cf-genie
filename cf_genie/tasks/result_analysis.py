import cf_genie.logger as logger
from cf_genie.models import SUPERVISED_MODELS
from cf_genie.models.model_analyisis import \
    get_acc_pandas_df_for_model_all_classes

logger.setup_applevel_logger(
    is_debug=False, file_name=__file__, simple_logs=True)


log = logger.get_logger(__name__)


def main():
    for model_class in SUPERVISED_MODELS:
        df = get_acc_pandas_df_for_model_all_classes(model_class)
        print(df)
    pass


if __name__ == '__main__':
    main()
