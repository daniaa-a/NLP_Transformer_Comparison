
import torch
import torch.nn as nn
import math


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, padding_idx=0):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_idx
        )

    def forward(self, x):
        return self.embedding(x)


class PositionalEmbedding(nn.Module):
    def __init__(self, max_length, embedding_dim):
        super().__init__()

        self.embedding = nn.Embedding(
            max_length,
            embedding_dim
        )

    def forward(self, x):
        positions = torch.arange(
            x.size(1),
            device=x.device
        ).unsqueeze(0)

        return self.embedding(positions)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):

        scores = torch.matmul(
            query,
            key.transpose(-2, -1)
        ) / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(
                mask == 0,
                torch.finfo(scores.dtype).min
            )

        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        output = torch.matmul(attention_weights, value)

        return output, attention_weights


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_heads,
        dropout=0.1
    ):
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_heads"
            )

        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)

        self.attention = ScaledDotProductAttention(dropout)

        self.output = nn.Linear(embedding_dim, embedding_dim)

    def split_heads(self, x):

        batch_size, seq_length, _ = x.shape

        x = x.view(
            batch_size,
            seq_length,
            self.num_heads,
            self.head_dim
        )

        return x.transpose(1, 2)

    def forward(self, x, mask=None):

        query = self.split_heads(self.query(x))
        key = self.split_heads(self.key(x))
        value = self.split_heads(self.value(x))

        attended, weights = self.attention(
            query,
            key,
            value,
            mask
        )

        attended = attended.transpose(1, 2).contiguous()

        batch_size, seq_length, _, _ = attended.shape

        attended = attended.view(
            batch_size,
            seq_length,
            self.num_heads * self.head_dim
        )

        return self.output(attended), weights


class FeedForward(nn.Module):
    def __init__(
        self,
        embedding_dim,
        hidden_dim,
        dropout=0.1
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, x):
        return self.network(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_heads,
        hidden_dim,
        dropout=0.1
    ):
        super().__init__()

        self.attention = MultiHeadAttention(
            embedding_dim,
            num_heads,
            dropout
        )

        self.feed_forward = FeedForward(
            embedding_dim,
            hidden_dim,
            dropout
        )

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):

        attention_output, _ = self.attention(x, mask)

        x = self.norm1(
            x + self.dropout(attention_output)
        )

        feed_forward_output = self.feed_forward(x)

        x = self.norm2(
            x + self.dropout(feed_forward_output)
        )

        return x


class ScratchTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        num_heads=4,
        hidden_dim=256,
        num_layers=2,
        max_length=256,
        num_classes=2,
        dropout=0.1,
        padding_idx=0
    ):
        super().__init__()

        self.padding_idx = padding_idx
        self.max_length = max_length

        self.token_embedding = TokenEmbedding(
            vocab_size,
            embedding_dim,
            padding_idx
        )

        self.position_embedding = PositionalEmbedding(
            max_length,
            embedding_dim
        )

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(
                embedding_dim,
                num_heads,
                hidden_dim,
                dropout
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(embedding_dim)

        self.classifier = nn.Linear(
            embedding_dim,
            num_classes
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):

        if input_ids.size(1) > self.max_length:
            raise ValueError(
                f"Input length exceeds maximum length of {self.max_length}"
            )

        padding_mask = (
            input_ids != self.padding_idx
        ).unsqueeze(1).unsqueeze(2)

        x = (
            self.token_embedding(input_ids)
            + self.position_embedding(input_ids)
        )

        x = self.dropout(x)

        for layer in self.encoder_layers:
            x = layer(x, padding_mask)

        x = self.final_norm(x)

        # Mask padding tokens before mean pooling
        valid_mask = (
            input_ids != self.padding_idx
        ).unsqueeze(-1).to(x.dtype)

        pooled = (
            (x * valid_mask).sum(dim=1)
            / valid_mask.sum(dim=1).clamp(min=1)
        )

        logits = self.classifier(pooled)

        return logits
