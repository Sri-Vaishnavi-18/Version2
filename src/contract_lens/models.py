from __future__ import annotations

import re
from collections import Counter

import torch
from torch import nn

TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+|[^\w\s]", re.IGNORECASE)
PAD_TOKEN = "<PAD>"
UNKNOWN_TOKEN = "<UNK>"


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


class Vocabulary:
    def __init__(self, token_to_id: dict[str, int]):
        self.token_to_id = token_to_id

    @classmethod
    def build(
        cls, texts: list[str], max_size: int = 20_000, min_frequency: int = 2
    ) -> "Vocabulary":
        counts = Counter(token for text in texts for token in tokenize(text))
        tokens = [
            token
            for token, count in counts.most_common()
            if count >= min_frequency and token not in {PAD_TOKEN, UNKNOWN_TOKEN}
        ][: max_size - 2]
        return cls({PAD_TOKEN: 0, UNKNOWN_TOKEN: 1, **{token: i + 2 for i, token in enumerate(tokens)}})

    def encode(self, text: str, max_length: int = 160) -> list[int]:
        tokens = tokenize(text)[:max_length]
        return [self.token_to_id.get(token, 1) for token in tokens] or [1]

    def to_dict(self) -> dict[str, int]:
        return self.token_to_id


class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embedding_dim: int = 96,
        hidden_dim: int = 64,
        dropout: float = 0.3,
        use_attention: bool = True,
    ):
        super().__init__()
        self.use_attention = use_attention
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.encoder = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        if use_attention:
            self.attention = nn.Linear(hidden_dim * 2, 1, bias=False)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(token_ids))
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        encoded_packed, (hidden, _) = self.encoder(packed)
        if self.use_attention:
            encoded, _ = nn.utils.rnn.pad_packed_sequence(
                encoded_packed, batch_first=True, total_length=token_ids.shape[1]
            )
            mask = torch.arange(token_ids.shape[1], device=token_ids.device)[None, :] < lengths[:, None]
            scores = self.attention(encoded).squeeze(-1).masked_fill(~mask, torch.finfo(encoded.dtype).min)
            weights = torch.softmax(scores, dim=1)
            pooled = torch.sum(encoded * weights.unsqueeze(-1), dim=1)
        else:
            pooled = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.classifier(self.dropout(pooled))
