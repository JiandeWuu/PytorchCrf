import torch
import torch.nn as nn

class Bilstm(torch.nn.Module):
    """BiLSTM Model 的初始化

    Args:
        nvocab (int): 詞庫 size
        ntoken (int): 輸出的類別 size
        ninp (int): 詞向量的 size
        nhid (int): 隱藏層 size
        nlayers (int): LSTM的層數
        dropout (float, optional): 如果不為零，則在除最後一層以外的每個LSTM層的輸出上引入一個Dropout層，其丟棄概率等於 dropout . Defaults to 0.0.
        batch_first (bool, optional): 如果為True，則將輸入和輸出張量提供為（批次，序列）. Defaults to True.
    """
    def __init__(self, nvocab: int, ntoken: int, ninp: int, nhid: int, nlayers: int, dropout: float=0.0, batch_first: bool=True):
        
        super(Bilstm, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(nvocab, ninp, padding_idx=0)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout, bidirectional=True, batch_first=batch_first)
        self.decoder = nn.Linear(nhid * 2, ntoken)

        self.init_weights()

        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        """初始化 embedding, liner 兩層的 weight
        """
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x: torch.tensor):
        """運算一次

        Args:
            x (torch.tensor): 詞 tensor。如果batch_first=True，x為（批次，序列），否則（序列，批次）。

        Returns:
            [torch.tensor]: 運算完的3D向量，多了一維類別機率分數。
            [torch.tensor]: LSTM層的Hidden weight。
        """
        batch_size = x.size()[0]

        emb = self.drop(self.encoder(x))
        
        hid = self.init_hidden(batch_size)
        output, hidden = self.rnn(emb, hid)
        output = self.drop(output)
        decoded = self.decoder(output)
        return decoded, hidden

    def init_hidden(self, bsz: int):
        """取得LSTM shape size 的初始化  weight

        Args:
            bsz (int): batch_size

        Returns:
            [tuple]: LSTM shape size 的初始化  weight
        """
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers * 2, bsz, self.nhid), weight.new_zeros(self.nlayers * 2, bsz, self.nhid))
    
    def predict(self, x: torch.tensor):
        """預測並輸出機率大的類別

        Args:
            x (torch.tensor): 詞 tensor。如果batch_first=True，input shape為（批次，序列），否則（序列，批次）。

        Returns:
            [torch.tensor]: shape 與 x 一樣，但是序列為類別序列。
        """
        decoded, _  = self.forward(x)
        _, output = decoded.max(dim=2)

        return output
