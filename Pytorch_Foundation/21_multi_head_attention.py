import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {device}\n")


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=16, num_heads=4):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)


        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape


        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)


        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)


        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)


        output = self.out_proj(attn_output)
        return output, attn_weights



torch.manual_seed(42)
batch_size, seq_len, embed_dim, num_heads = 1, 4, 16, 4

x = torch.randn(batch_size, seq_len, embed_dim).to(device)
mha = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads).to(device)

output, weights = mha(x)

print(f"Input Tensor Shape:              {x.shape}")
print(f"Multi-Head Output Shape:         {output.shape}")
print(f"Attention Weights Shape (B,H,S,S): {weights.shape}")
print(f"\nHead 0 Attention Weights Matrix:\n{weights[0, 0]}")

