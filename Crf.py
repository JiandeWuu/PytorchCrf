import time

import torch
import torch.nn as nn

class Crf(nn.Module):
    """CRF Model

    Args:
        tagset_size (int): 類別的數量。
        batch_first (bool, optional): 如果為True，則將輸入和輸出張量提供為（批次，序列）. Defaults to False.
    """
    def __init__(self, tagset_size: int, batch_first: bool=False):
        
        super(Crf, self).__init__()
        self.tagset_size = tagset_size
        self.batch_first = batch_first

        self.start_transitions = nn.Parameter(torch.zeros(self.tagset_size))
        self.end_transitions = nn.Parameter(torch.zeros(self.tagset_size))
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))

    def _forward_alg(self, feats: torch.tensor, mask: torch.tensor):
        """計算所有可能的路線的總和分數

        Args:
            feats (torch.tensor): 分數 tensor。
            mask (torch.tensor): 遮罩 tensor。

        Returns:
            [torch.tensor]: 總和的路線分數 tensor。
        """
        seq_length = feats.size(0)

        score = self.start_transitions + feats[0]
        for i in range(1, seq_length):
            broadcast_score = score.unsqueeze(2)

            broadcast_feats = feats[i].unsqueeze(1)

            next_score = broadcast_score + self.transitions + broadcast_feats

            next_score = torch.logsumexp(next_score, dim=1)

            score = torch.where(mask[i].byte().unsqueeze(1), next_score, score)
        score += self.end_transitions
        score = torch.logsumexp(score, dim=1)
        
        return score

    def _score_sentence(self, feats: torch.tensor, tags: torch.tensor, mask: torch.tensor):
        """計算正確的路線分數

        Args:
            feats (torch.tensor): 特徵分數 tensor。
            tags (torch.tensor): 類別 tensor。
            mask (torch.tensor): 遮罩 tensor。

        Returns:
            [torch.tensor]: 正確的路線分數 tensor。
        """
        # Gives the score of a provided tag sequence
        seq_length = tags.size(0)
        batch_size = tags.size(1)

        score = self.start_transitions[tags[0]]
        score += feats[0, torch.arange(batch_size), tags[0]]
        for i in range(1, seq_length):
            score += feats[i, torch.arange(batch_size), tags[i]] * mask[i]
            score += self.transitions[tags[i - 1], tags[i]] * mask[i]
        
        seq_ends = mask.long().sum(dim=0) - 1
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        score += self.end_transitions[last_tags]
        return score
    
    def predict(self, feats: torch.tensor, mask: torch.tensor=None):
        """解碼特徵分數 tensor 成類別 tensor。

        Args:
            feats (torch.tensor): 特徵分數 tensor。
            mask (torch.tensor, optional): 遮罩 tensor，遮罩值為0，否則為1. Defaults to None.

        Returns:
            [list]: 類別 list。
        """
        if mask is None:
            mask = feats.new_ones(feats.size()[:2], dtype=torch.uint8)
        if self.batch_first:
            feats = feats.transpose(0, 1)
            mask = mask.transpose(0, 1)
        return self._viterbi_decode(feats, mask)

    def _viterbi_decode(self, feats: torch.tensor, mask: torch.tensor):
        """解碼特徵分數 tensor 成類別 tensor。

        Args:
            feats (torch.tensor): 特徵分數 tensor。
            mask (torch.tensor): 遮罩 tensor。

        Returns:
            [list]: 類別 list。
        """
        backpointers = []

        seq_length, batch_size, _ = feats.size()
        score = self.start_transitions + feats[0]
        for i in range(1, seq_length):
            broadcast_score = score.unsqueeze(2)

            broadcast_feats = feats[i].unsqueeze(1)

            next_score = broadcast_score + self.transitions + broadcast_feats

            next_score, indices = next_score.max(dim=1)

            score = torch.where(mask[i].byte().unsqueeze(1), next_score, score)

            backpointers.append(indices.tolist())
        score += self.end_transitions

        seq_ends = mask.long().sum(dim=0) - 1

        best_tags_list = []
        for idx in range(batch_size):
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]
            for hist in reversed(backpointers[:seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag)

            best_tags.reverse()
            best_tags_list.append(best_tags)
        return best_tags_list
     
    def forward(self, feats: torch.tensor, tags: torch.tensor, mask: torch.tensor=None):
        """運算，在這專案中是使用由上一層神經層傳遞下來的特徵分數tensor，

        Args:
            feats (torch.tensor): 特徵分數 tensor。
            tags (torch.tensor): 類別 tensor。
            mask (torch.tensor, optional): 遮罩 tensor，遮罩值為0，否則為1. Defaults to None.

        Returns:
            [torch.tensor]: Loss tensor。
        """
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8, device=feats.get_device())
        
        if self.batch_first:
            feats = feats.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        forward_score = self._forward_alg(feats, mask)
        gold_score = self._score_sentence(feats, tags, mask)

        return forward_score - gold_score

