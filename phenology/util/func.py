import numpy as np


def round_partial(value, resolution):
    # Thanks to https://stackoverflow.com/questions/8118679/python-rounding-by-quarter-intervals
    return round(value / resolution) * resolution


def create_left_mask(length: int, ix: int) -> np.ndarray:
    mask = np.arange(length)
    mask = np.where(mask >= ix, 1, 0)
    return mask


if __name__ == '__main__':

    print(create_left_mask(10, 4))
