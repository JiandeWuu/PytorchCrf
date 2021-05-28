import time

import torch
from .Bilstm import Bilstm
from .Crf import Crf

class BilstmCrf(torch.nn.Module):
    """BiLSTM + CRF Model

    Args:
        nvocab (int): 詞庫 size
        ntoken (int): 輸出的類別 size
        ninp (int): 詞向量的 size
        nhid (int): 隱藏層 size
        nlayers (int): LSTM的層數
        dropout (float, optional): 如果不為零，則在除最後一層以外的每個LSTM層的輸出上引入一個Dropout層，其丟棄概率等於 dropout . Defaults to 0.0.
        batch_first (bool, optional): 如果為True，則將輸入和輸出張量提供為（批次，序列） . Defaults to True.
    """
    def __init__(self, nvocab: int, ntoken: int, ninp: int, nhid: int, nlayers: int, dropout: float=0.0, batch_first: bool=True):
        super(BilstmCrf, self).__init__()
        self.bilstm = Bilstm(nvocab, ntoken, ninp, nhid, nlayers, dropout=dropout, batch_first=batch_first)
        self.crf = Crf(ntoken, batch_first=batch_first)
        self.tag_size = ntoken

    def forward(self, x: torch.tensor, tag: torch.tensor, mask: torch.tensor=None):
        """[summary]

        Args:
            x (torch.tensor): 詞 tensor。批次好的 tensor。如果batch_first=True，input為（批次，序列），否則（序列，批次）。
            tag (torch.tensor): 類別 tensor。批次好的 tensor。如果batch_first=True，input為（批次，序列），否則（序列，批次）。
            mask (torch.tensor, optional): 遮罩 tensor，遮罩值為0，否則為1. Defaults to None.

        Returns:
            [torch.tensor]: [description]
        """
        batch_size = x.size()[0]

        lstm_out, _ = self.bilstm(x)

        loss = self.crf(lstm_out, tag, mask=mask)

        return torch.sum(loss) / batch_size

    def predict(self, x: torch.tensor, mask: torch.tensor=None):
        """預測並輸出機率大的類別

        Args:
            x (torch.tensor): 詞 tensor。
            mask (torch.tensor, optional): 遮罩 tensor，遮罩值為0，否則為1. Defaults to None.

        Returns:
            [list]: 類別 list。
        """
        lstm_out, _ = self.bilstm(x)
        tag_seq = self.crf.decode(lstm_out, mask=mask)

        return tag_seq
        