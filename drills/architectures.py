import torch
import torch.nn as nn
from typing import Optional
from transformer_block import TransformerBlock, TransformerDecoderBlock
from attention import make_causal_mask
from positional import LearnedPositionalEmbedding


# ---------------------------------------------------------------------------
# Encoder (BERT-style)
# ---------------------------------------------------------------------------

class TransformerEncoder(nn.Module):
    """Stack of transformer blocks with no causal mask — bidirectional attention.
    Every token can attend to every other token.
    Used in: BERT, RoBERTa, encoder half of T5.
    """

    def __init__(self, d_model: int, num_heads: int, num_layers: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, expansion=expansion, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x


# ---------------------------------------------------------------------------
# Decoder (GPT-style, no cross-attention)
# ---------------------------------------------------------------------------

class TransformerDecoder(nn.Module):
    """Stack of transformer blocks with a causal mask — autoregressive.
    Each token can only attend to itself and previous tokens.
    Used as the core of decoder-only models like GPT.
    """

    def __init__(self, d_model: int, num_heads: int, num_layers: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, expansion=expansion, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        T = x.size(1)
        causal = make_causal_mask(T, device=x.device)
        combined = causal if mask is None else causal.unsqueeze(0) & mask
        for layer in self.layers:
            x = layer(x, mask=combined)
        return x


# ---------------------------------------------------------------------------
# Encoder-Decoder (original transformer)
# ---------------------------------------------------------------------------

class EncoderDecoder(nn.Module):
    """Full encoder-decoder transformer (Vaswani et al. 2017).
    Encoder: bidirectional over source sequence.
    Decoder: causal self-attention + cross-attention to encoder outputs.
    Used in: T5, BART, original MT transformer.
    """

    def __init__(self, d_model: int, num_heads: int, num_layers: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        self.encoder_layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, expansion=expansion, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderBlock(d_model, num_heads, expansion=expansion, dropout=dropout)
            for _ in range(num_layers)
        ])

    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = src
        for layer in self.encoder_layers:
            x = layer(x, mask=src_mask)
        return x

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        T = tgt.size(1)
        causal = make_causal_mask(T, device=tgt.device)
        self_mask = causal if tgt_mask is None else causal.unsqueeze(0) & tgt_mask
        x = tgt
        for layer in self.decoder_layers:
            x = layer(x, memory, self_mask=self_mask, cross_mask=memory_mask)
        return x

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        memory = self.encode(src, src_mask)
        return self.decode(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)


# ---------------------------------------------------------------------------
# GPT (decoder-only)
# ---------------------------------------------------------------------------

class GPT(nn.Module):
    """Decoder-only transformer language model (GPT-style).
    token embedding + learned positional embedding → causal transformer stack → LM head.
    LM head weight is tied to the token embedding matrix (reduces parameters).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        max_seq_len: int,
        expansion: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = LearnedPositionalEmbedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.decoder = TransformerDecoder(d_model, num_heads, num_layers, expansion=expansion, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight                              # weight tying

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (B, T) token indices
        Returns:
            logits: (B, T, vocab_size)
        """
        x = self.drop(self.pos_emb(self.token_emb(input_ids)))
        x = self.decoder(x)
        x = self.norm(x)
        return self.lm_head(x)


# ---------------------------------------------------------------------------
# BERT (encoder-only)
# ---------------------------------------------------------------------------

class BERT(nn.Module):
    """Encoder-only transformer (BERT-style).
    token embedding + segment embedding + learned positional embedding → bidirectional encoder.
    Outputs contextual representations — no LM head here (add task-specific heads on top).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        max_seq_len: int,
        num_segments: int = 2,
        expansion: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.segment_emb = nn.Embedding(num_segments, d_model)
        self.pos_emb = LearnedPositionalEmbedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.encoder = TransformerEncoder(d_model, num_heads, num_layers, expansion=expansion, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        input_ids: torch.Tensor,
        segment_ids: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids:   (B, T) token indices
            segment_ids: (B, T) 0 or 1 for sentence A/B (optional)
            mask:        (B, T) padding mask (optional)
        Returns:
            (B, T, d_model) contextual embeddings
        """
        x = self.token_emb(input_ids)
        if segment_ids is not None:
            x = x + self.segment_emb(segment_ids)
        x = self.drop(self.pos_emb(x))
        x = self.encoder(x, mask=mask)
        return self.norm(x)
