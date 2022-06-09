import multiprocessing


def get_num_of_cores() -> int:
    return multiprocessing.cpu_count()
