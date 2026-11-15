import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device {device}")

batch_size = 1
seq_len = 4
embed_dim = 16
d_k = 16

torch.manual_seed(42)

X = torch.randn(embed_dim,seq_len,d_k).to(device)

W_q = nn.Linear(embed_dim,d_k,bias=False).to(device)
W_k = nn.Linear(embed_dim,d_k,bias=False).to(device)
W_v = nn.Linear(embed_dim,d_k,bias=False).to(device)

Q = W_q(X)
K = W_k(X)
V = W_v(X)

print(f"Query Matrix {Q.shape}")

K_T = K.transpose(-2,-1)

scores = torch.matmul(Q,K_T)
print(scores[0])

scale = torch.sqrt(torch.tensor(d_k, dtype=torch.float32)).to(device)
scaled_scores = scores / scale

attention_weights = F.softmax(scaled_scores,dim=-1)
print(attention_weights[0])

print(f"Rows Sums (Verification) {attention_weights.sum(dim=-1)} ")

output = torch.matmul(attention_weights,V)
print(output.shape)