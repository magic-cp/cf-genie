"""
Script to cleanup the dataset. Performs preprocessing over CF statements, and also remapping of tags to "tag groups"
"""
import cf_genie.logger as logger
import cf_genie.utils as utils

logger.setup_applevel_logger(
    is_debug=False, file_name=__file__, simple_logs=True)


log = logger.get_logger(__name__)


def main(*args):

    df = utils.read_raw_dataset()


if __name__ == '__main__':
    main()
