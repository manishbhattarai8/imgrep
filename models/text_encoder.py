import torch.nn as nn
import torch

class TransformerTextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_heads=4, num_layers=2, max_len=40):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_dim = embed_dim

    def forward(self, x):
        # x: (B, T)
        B, T = x.size()
        pos = torch.arange(0, T).unsqueeze(0).to(x.device)  # (1, T)
        x = self.embedding(x) + self.pos_embedding(pos)     # (B, T, D)
        x = x.permute(1, 0, 2)                              # (T, B, D)
        x = self.transformer(x)                             # (T, B, D)
        x = x.mean(dim=0)                                   # (B, D)
        return x