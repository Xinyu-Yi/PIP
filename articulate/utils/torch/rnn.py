r"""
    Utils for RNN including networks, datasets, and loss wrappers.
"""


__all__ = ['RNNLossWrapper', 'RNNDataset', 'RNNWithInitDataset', 'RNN', 'RNNWithInit']


import os
import torch.utils.data
from torch.nn.functional import relu
from torch.nn.utils.rnn import *


class RNNLossWrapper:
    r"""
    Loss wrapper for `articulate.utils.torch.RNN`.
    """
    def __init__(self, loss_fn):
        self.loss_fn = loss_fn

    def __call__(self, y_pred, y_true):
        return self.loss_fn(torch.cat(y_pred), torch.cat(y_true))


class RNNDataset(torch.utils.data.Dataset):
    r"""
    Dataset for `articulate.utils.torch.RNN`.
    """
    def __init__(self, data: list, label: list, split_size=-1, augment_fn=None, device=None):
        r"""
        Init an RNN dataset.

        Notes
        -----
        Get the dataloader by torch.utils.data.DataLoader(dataset, **collate_fn=RNNDataset.collate_fn**)

        If `split_size` is positive, `data` and `label` will be split to lists of small sequences whose lengths
        are not larger than `split_size`. Otherwise, it has no effects.

        If `augment_fn` is not None, `data` item will be augmented like `augment_fn(data[i])` in `__getitem__`.
        Otherwise, it has no effects.

        Args
        -----
        :param data: A list that contains sequences(tensors) in shape [num_frames, input_size].
        :param label: A list that contains sequences(tensors) in shape [num_frames, output_size].
        :param split_size: If positive, data and label will be split to list of small sequences.
        :param augment_fn: If not None, data item will be augmented in __getitem__.
        :param device: The loaded data is finally copied to the device. If None, the device of data[0] is used.
        """
        assert len(data) == len(label) and len(data) != 0
        if split_size > 0:
            self.data, self.label = [], []
            for td, tl in zip(data, label):
                self.data.extend(td.split(split_size))
                self.label.extend(tl.split(split_size))
        else:
            self.data = data
            self.label = label
        self.augment_fn = augment_fn
        self.device = device if device is not None else data[0].device

    def __getitem__(self, i):
        data = self.data[i] if self.augment_fn is None else self.augment_fn(self.data[i])
        label = self.label[i]
        return data.to(self.device), label.to(self.device)

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(x):
        r"""
        [[seq0, label0], [seq1, label1], [seq2, label2]] -> [[seq0, seq1, seq2], [label0, label1, label2]]
        """
        return list(zip(*x))


class RNNWithInitDataset(RNNDataset):
    r"""
    The same as `RNNDataset`. Used for `RNNWithInit`.
    """
    def __init__(self, data: list, label: list, split_size=-1, augment_fn=None, device=None):
        super(RNNWithInitDataset, self).__init__(data, label, split_size, augment_fn, device)

    def __getitem__(self, i):
        data, label = super(RNNWithInitDataset, self).__getitem__(i)
        return (data, label[0]), label


class RNN(torch.nn.Module):
    r"""
    An RNN net including a linear input layer, an RNN, and a linear output layer.
    """
    def __init__(self, input_size: int, output_size: int, hidden_size: int, num_rnn_layer: int,
                 rnn_type='lstm', bidirectional=False, dropout=0., load_weight_file: str = None):
        r"""
        Init an RNN.

        :param input_size: Input size.
        :param output_size: Output size.
        :param hidden_size: Hidden size for RNN.
        :param num_rnn_layer: Number of RNN layers.
        :param rnn_type: Select from 'rnn', 'lstm', 'gru'.
        :param bidirectional: Whether if the RNN is bidirectional.
        :param dropout: Dropout after the input linear layer and in the rnn.
        :param load_weight_file: If not None and exists, weights will be loaded.
        """
        super().__init__()
        self.rnn = getattr(torch.nn, rnn_type.upper())(hidden_size, hidden_size, num_rnn_layer,
                                                       bidirectional=bidirectional, dropout=dropout)
        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size * (2 if bidirectional else 1), output_size)
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()

        if load_weight_file and os.path.exists(load_weight_file):
            self.load_state_dict(torch.load(load_weight_file, map_location=torch.device('cpu')))
            self.eval()

    def forward(self, x, init=None):
        r"""
        Forward.

        :param x: A list in length [batch_size] which contains tensors in shape [num_frames, input_size].
        :param init: Initial hidden states.
        :return: A list in length [batch_size] which contains tensors in shape [num_frames, output_size].
        """
        length = [_.shape[0] for _ in x]
        x = self.dropout(relu(self.linear1(pad_sequence(x))))
        x = self.rnn(pack_padded_sequence(x, length, enforce_sorted=False), init)[0]
        x = self.linear2(pad_packed_sequence(x)[0])
        return [x[:l, i].clone() for i, l in enumerate(length)]


class RNNWithInit(RNN):
    r"""
    RNN with the initial hidden states regressed from the first output.
    """
    def __init__(self, input_size: int, output_size: int, hidden_size: int, num_rnn_layer: int,
                 rnn_type='lstm', bidirectional=False, dropout=0., load_weight_file: str = None):
        r"""
        Init an RNNWithInit net.

        :param input_size: Input size.
        :param output_size: Output size.
        :param hidden_size: Hidden size for RNN.
        :param num_rnn_layer: Number of RNN layers.
        :param rnn_type: Select from 'rnn', 'lstm', 'gru'.
        :param bidirectional: Whether if the RNN is bidirectional.
        :param dropout: Dropout after the input linear layer and in the rnn.
        :param load_weight_file: If not None and exists, weights will be loaded.
        """
        assert rnn_type == 'lstm' and bidirectional is False
        super().__init__(input_size, output_size, hidden_size, num_rnn_layer, rnn_type, bidirectional, dropout)

        self.init_net = torch.nn.Sequential(
            torch.nn.Linear(output_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size * num_rnn_layer),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size * num_rnn_layer, 2 * (2 if bidirectional else 1) * num_rnn_layer * hidden_size)
        )

        if load_weight_file and os.path.exists(load_weight_file):
            self.load_state_dict(torch.load(load_weight_file, map_location=torch.device('cpu')))
            self.eval()

    def forward(self, x, _=None):
        r"""
        Forward.

        :param x: A list in length [batch_size] which contains 2-tuple
                  (Tensor[num_frames, input_size], Tensor[output_size]).
        :param _: Not used.
        :return: A list in length [batch_size] which contains tensors in shape [num_frames, output_size].
        """
        x, x_init = list(zip(*x))
        nd, nh = self.rnn.num_layers * (2 if self.rnn.bidirectional else 1), self.rnn.hidden_size
        h, c = self.init_net(torch.stack(x_init)).view(-1, 2, nd, nh).permute(1, 2, 0, 3)
        return super(RNNWithInit, self).forward(x, (h, c))


class CycleRNN(torch.nn.Module):  # todo: dropout order, not very well, .predict() with hidden
    r"""
    A cycle RNN net including a linear input layer, an RNN, and a linear output layer.
    Each previous estimation is fed into the network as the input for the next time step.
    """
    def __init__(self, input_size: int, output_size: int, hidden_size: int, num_rnn_layer: int,
                 rnn_type='lstm', bidirectional=False, dropout_input=0., dropout_rnn=0.,
                 pred_weight=1, load_weight_file=None):
        r"""
        Init a CycleRNN.

        :param input_size: Input size (after concatenating the estimation).
        :param output_size: Output size.
        :param hidden_size: Hidden size for RNN.
        :param num_rnn_layer: Number of RNN layers.
        :param rnn_type: Select from 'rnn', 'lstm', 'gru'.
        :param bidirectional: Whether if the RNN is bidirectional. Only support unidirectional RNN.
        :param dropout_input: Dropout rate on the input.
        :param dropout_rnn: Dropout rate on the rnn.
        :param pred_weight: The lerp weight for previous estimation / ground truth.
        :param load_weight_file: If not None and exists, weights will be loaded.
        """
        super(CycleRNN, self).__init__()
        assert bidirectional is False
        self.rnn = getattr(torch.nn, rnn_type.upper())(hidden_size, hidden_size, num_rnn_layer, dropout=dropout_rnn)
        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size * (2 if bidirectional else 1), output_size)
        self.dropout = torch.nn.Dropout(dropout_input) if dropout_input > 0 else torch.nn.Identity()
        self.output_size = output_size
        self.pred_weight = pred_weight

        if load_weight_file and os.path.exists(load_weight_file):
            self.load_state_dict(torch.load(load_weight_file))

    def _lerp(self, pred, true):
        return pred * self.pred_weight + true * (1 - self.pred_weight)

    def forward(self, x, hidden=None):
        r"""
        Forward. The ground truth output t-1 should be concatenated after each input t.

        :param x: A list in length `batch_size`, which contains tensors in shape [num_frames, input_size].
        :param hidden: Initial hidden state.
        :return: A list in length `batch_size`, which contains tensors in shape [num_frames, output_size].
        """
        length = [_.shape[0] for _ in x]
        x = pad_sequence(x).clone()
        result = [x[0, :, -self.output_size:]]

        for xi in x:
            xi[:, -self.output_size:] = self._lerp(result[-1].detach(), xi[:, -self.output_size:].detach())
            xi = self.dropout(relu(self.linear1(xi.clone())))
            xi, hidden = self.rnn(xi.unsqueeze(0), hidden)
            xi = self.linear2(xi.squeeze(0))
            result.append(xi)
        x = torch.stack(result[1:])

        return [x[:l, i].clone() for i, l in enumerate(length)]
