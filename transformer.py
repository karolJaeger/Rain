import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, in_ch, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)

        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.0001)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(TransformerBlock, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.input_norm = nn.LayerNorm(embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.linear = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.Dropout(p=0.0001)
                )
        self.out_norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.GELU(),
                nn.Linear(embed_dim * 2, embed_dim),
                nn.Dropout(p=0.0001)
                )

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, inp):
        batch_size, seq_length, embed_dim = inp.size()

        x = self.input_norm(inp)
        query = self.split_heads(self.query(x), batch_size)
        key = self.split_heads(self.key(x), batch_size)
        value = self.split_heads(self.value(x), batch_size)

        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = torch.softmax(scores, dim=-1)

        out = torch.matmul(attention_weights, value)
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_length, embed_dim)
        out = self.linear(out)
        x = x + out
        x = self.out_norm(x)
        out = self.mlp(x)
        out = out + x

        return out


if __name__ == '__main__':
    batch_size = 2
    img_size = 256, 256
    channel = 3
    patch_size = 1
    vector_len = 1
    num_heads = 1

    patch_embedding = PatchEmbedding(patch_size=patch_size, in_ch=channel, embed_dim=vector_len)
    pos_encoder = PositionalEncoding(d_model=vector_len, max_len=patch_size ** 2)
    attention = TransformerBlock(embed_dim=vector_len, num_heads=num_heads)

    img_input = torch.randn(batch_size, channel, *img_size)
    patch_image = patch_embedding(img_input)
    pos_image = pos_encoder(patch_image)
    output = attention(pos_image)






