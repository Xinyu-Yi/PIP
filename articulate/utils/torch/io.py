r"""
    Utils for txt save/load.
"""


__all__ = ['load_txt_mat', 'save_txt_mat']


import torch


def load_txt_mat(file, sep=','):
    r"""
    Load a matrix in txt where each line is the row and each entry is separated by `sep`.

    :param file: txt file name.
    :param sep: Sep char for each entry in a row.
    :return: A 2d tensor.
    """
    with open(file, 'r') as f:
        data = torch.tensor([[float(s) for s in line.split(sep) if s != '\n'] for line in f.readlines()])
    return data


def save_txt_mat(mat, file, sep=','):
    r"""
    Save a matrix in txt where each line is the row and each entry is separated by `sep`.

    :param mat: The 2d tensor to save.
    :param file: txt file name.
    :param sep: Sep char for each entry in a row.
    """
    with open(file, 'w') as f:
        f.write('\n'.join([sep.join(['%.6f' % c for c in r]) for r in mat]))
