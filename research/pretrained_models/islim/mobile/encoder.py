"""
Visual Encoder for iSign keypoints
Processes temporal keypoint sequences into video embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TemporalConv1D(nn.Module):
    """Temporal 1D convolution for motion features"""

    def __init__(self, in_channels, out_channels, kernel_size=5, dropout=0.1):
        super().__init__()

        padding = kernel_size // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, C, T)
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class VisualEncoder(nn.Module):
    """
    Encodes keypoint sequences into video embeddings

    Architecture:
    1. TemporalConv1D layers for motion features
    2. Transformer encoder for temporal modeling
    3. Attention pooling for sign boundary detection
    4. Projection to T5 dimension
    """

    def __init__(
        self,
        keypoint_dim=4,  # x, y, z, visibility
        num_keypoints=94,
        temporal_channels=[376, 256, 256],  # Fixed: 376 = 94*4
        temporal_kernel=5,
        transformer_embed_dim=512,
        transformer_num_heads=8,
        transformer_num_layers=4,
        transformer_dropout=0.1,
        output_dim=512,
    ):
        super().__init__()

        self.keypoint_dim = keypoint_dim
        self.num_keypoints = num_keypoints
        self.transformer_embed_dim = transformer_embed_dim

        # Flatten keypoints: (B, T, 94, 4) -> (B, T, 376)
        flat_dim = num_keypoints * keypoint_dim  # 376
        self.keypoint_proj = nn.Linear(flat_dim, temporal_channels[0])

        # Temporal convolution layers
        self.temporal_convs = nn.ModuleList()
        in_ch = temporal_channels[0]
        for out_ch in temporal_channels[1:]:
            self.temporal_convs.append(
                TemporalConv1D(in_ch, out_ch, temporal_kernel, transformer_dropout)
            )
            in_ch = out_ch

        # Project to transformer dimension
        self.transformer_proj = nn.Linear(temporal_channels[-1], transformer_embed_dim)

        # Positional encoding
        self.pos_embedding = nn.Parameter(
            torch.randn(1, 1000, transformer_embed_dim) * 0.1
        )

        # Transformer encoder with Pre-Layer Norm
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_embed_dim,
            nhead=transformer_num_heads,
            dim_feedforward=transformer_embed_dim * 4,
            dropout=transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-Layer Norm - improves training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_num_layers
        )

        # Attention pooling
        self.attention_query = nn.Parameter(
            torch.randn(1, 1, transformer_embed_dim) * 0.1
        )
        self.attention_proj = nn.Linear(transformer_embed_dim, 1)

        # Output projection
        self.output_proj = nn.Linear(transformer_embed_dim, output_dim)

        self.layer_norm = nn.LayerNorm(transformer_embed_dim)

    def forward(self, keypoints, mask=None):
        """
        Args:
            keypoints: (B, T, 94, 4) or (B, 4, 94, T)

        Returns:
            embedding: (B, output_dim)
        """
        # Handle different input formats
        if keypoints.dim() == 4 and keypoints.shape[1] == 4:
            # (B, 4, 94, T) -> (B, T, 94, 4)
            keypoints = keypoints.permute(0, 3, 2, 1)

        B, T, K, D = keypoints.shape  # Batch, Time, Keypoints, Dims

        # Flatten keypoints: (B, T, 94*4)
        x = keypoints.reshape(B, T, -1)

        # Project keypoints
        x = self.keypoint_proj(x)  # (B, T, temporal_channels[0])

        # Transpose for Conv1d: (B, C, T)
        x = x.transpose(1, 2)

        # Temporal convolutions
        for conv in self.temporal_convs:
            x = conv(x)

        # Transpose back: (B, T, C)
        x = x.transpose(1, 2)

        # Project to transformer dimension
        x = self.transformer_proj(x)  # (B, T, transformer_embed_dim)

        # Add positional encoding
        if x.shape[1] <= self.pos_embedding.shape[1]:
            x = x + self.pos_embedding[:, : x.shape[1], :]

        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=mask)
        x = self.layer_norm(x)

        # Attention pooling
        # Learnable query: (1, 1, D) -> (B, 1, D)
        query = self.attention_query.expand(B, -1, -1)

        # Compute attention scores
        attn_scores = self.attention_proj(x)  # (B, T, 1)

        # Mask padding
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(-1), float("-inf"))

        # Softmax over time
        attn_weights = F.softmax(attn_scores, dim=1)  # (B, T, 1)

        # Weighted sum
        pooled = (x * attn_weights).sum(dim=1)  # (B, D)

        # Output projection
        output = self.output_proj(pooled)  # (B, output_dim)

        return output

    def get_sequence_output(self, keypoints, mask=None):
        """Get per-timestep output for debugging"""
        # Same as forward but return full sequence
        if keypoints.dim() == 4 and keypoints.shape[1] == 4:
            keypoints = keypoints.permute(0, 3, 2, 1)

        B, T, K, D = keypoints.shape
        x = keypoints.reshape(B, T, -1)
        x = self.keypoint_proj(x)
        x = x.transpose(1, 2)

        for conv in self.temporal_convs:
            x = conv(x)

        x = x.transpose(1, 2)
        x = self.transformer_proj(x)

        if x.shape[1] <= self.pos_embedding.shape[1]:
            x = x + self.pos_embedding[:, : x.shape[1], :]

        x = self.transformer(x, src_key_padding_mask=mask)

        return x  # (B, T, D)


class KeypointToVideoEncoder(nn.Module):
    """
    Wrapper that handles variable-length keypoint sequences
    and prepares input for T5
    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.max_frames = config.get("max_frames", 128)

        self.encoder = VisualEncoder(
            keypoint_dim=config.get("keypoint_dim", 4),
            num_keypoints=config.get("num_keypoints", 94),
            temporal_channels=config.get("temporal_channels", [376, 256, 256]),
            temporal_kernel=config.get("temporal_kernel", 5),
            transformer_embed_dim=config.get("transformer_embed_dim", 512),
            transformer_num_heads=config.get("transformer_num_heads", 8),
            transformer_num_layers=config.get("transformer_num_layers", 4),
            transformer_dropout=config.get("transformer_dropout", 0.1),
            output_dim=config.get("output_dim", 512),
        )

    def forward(self, keypoints, padding_mask=None):
        """
        Args:
            keypoints: (B, T, 94, 4) or (B, 4, 94, T)

        Returns:
            video_embedding: (B, output_dim) or (B, T, output_dim) if return_sequence=True
        """
        # Handle different input formats
        # Dataset outputs (B, 4, 94, T), but we expect (B, T, 94, 4)
        if keypoints.dim() == 4 and keypoints.shape[1] == 4:
            # (B, 4, 94, T) -> (B, T, 94, 4)
            keypoints = keypoints.permute(0, 3, 2, 1)

        # Truncate or pad to max_frames
        B, T, K, D = keypoints.shape

        if T > self.max_frames:
            # Uniform sampling
            indices = (
                torch.linspace(0, T - 1, self.max_frames).long().to(keypoints.device)
            )
            keypoints = keypoints[:, indices, :, :]
        elif T < self.max_frames:
            # Pad
            pad_size = self.max_frames - T
            padding = torch.zeros(B, pad_size, K, D).to(keypoints.device)
            keypoints = torch.cat([keypoints, padding], dim=1)
            if padding_mask is not None:
                pad_mask = torch.ones(B, pad_size).to(padding_mask.device)
                padding_mask = torch.cat([padding_mask, pad_mask], dim=1)

        # Encode
        embedding = self.encoder(keypoints, padding_mask)

        return embedding

    def get_sequence_output(self, keypoints, padding_mask=None):
        """
        Get full sequence output for T5 encoder

        Args:
            keypoints: (B, T, 94, 4) or (B, 4, 94, T)

        Returns:
            sequence_output: (B, T, output_dim)
        """
        # Handle different input formats
        if keypoints.dim() == 4 and keypoints.shape[1] == 4:
            keypoints = keypoints.permute(0, 3, 2, 1)

        B, T, K, D = keypoints.shape

        if T > self.max_frames:
            indices = (
                torch.linspace(0, T - 1, self.max_frames).long().to(keypoints.device)
            )
            keypoints = keypoints[:, indices, :, :]
        elif T < self.max_frames:
            pad_size = self.max_frames - T
            padding = torch.zeros(B, pad_size, K, D).to(keypoints.device)
            keypoints = torch.cat([keypoints, padding], dim=1)

        # Get sequence output from encoder
        sequence_output = self.encoder.get_sequence_output(keypoints, padding_mask)

        # Project to output dim
        sequence_output = self.encoder.output_proj(sequence_output)

        return sequence_output


if __name__ == "__main__":
    # Test the encoder
    config = {
        "keypoint_dim": 4,
        "num_keypoints": 94,
        "temporal_channels": [376, 256, 256],  # 376 = 94*4
        "transformer_embed_dim": 512,
        "transformer_num_heads": 8,
        "transformer_num_layers": 4,
        "output_dim": 512,
        "max_frames": 128,
    }

    model = KeypointToVideoEncoder(config)

    # Test input
    batch_size = 2
    time_steps = 64

    keypoints = torch.randn(batch_size, time_steps, 94, 4)

    with torch.no_grad():
        output = model(keypoints)

    print(f"Input shape: {keypoints.shape}")
    print(f"Output shape: {output.shape}")

    # Test with variable lengths
    for T in [32, 64, 128, 256]:
        keypoints = torch.randn(batch_size, T, 94, 4)
        with torch.no_grad():
            output = model(keypoints)
        print(f"T={T}: output shape = {output.shape}")
