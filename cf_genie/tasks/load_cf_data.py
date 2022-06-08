import cf_genie.logger as logger
from cf_genie.utils import load_contests, load_problems


logger.setup_applevel_logger(
    is_debug=False, file_name=__file__, simple_logs=False)

log = logger.get_logger(__name__)

def main():
    log.info('Loading problems...')
    load_problems()
    log.info('Loading contests...')
    load_contests()
    log.info('Done!')

if __name__ == '__main__':
    main()
