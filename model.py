# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import torch
import torch.nn as nn
from torch.nn import functional as F
import math

torch.manual_seed(1337)


# @torch.jit.script # good to enable when not using torch.compile, disable when using (our default)
def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, n_embed, block_size, dropout):
        super().__init__()

        self.n_heads = n_heads
        self.n_embed = n_embed
        self.dropout = dropout
        self.attention = nn.Linear(n_embed, 3 * n_embed)
        self.projection = nn.Linear(n_embed, n_embed)
        self.attention_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)

        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                                        .view(1, 1, block_size, block_size))

    def _reshape_attn(self, batch, time, channels, matrix):
        return matrix.view(batch, time, self.n_heads, channels // self.n_heads).transpose(1, 2)

    def forward(self, x):
        batch, time, channels = x.size()

        query, key, value = self.attention(x).split(self.n_embed, dim=2)
        key = self._reshape_attn(batch, time, channels, key)
        query = self._reshape_attn(batch, time, channels, query)
        value = self._reshape_attn(batch, time, channels, value)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True)
        else:
            # manual implementation of attention
            weights = (query @ key.transpose(-2, -1)) * (1.0 / math.sqrt(key.size(-1)))
            weights = weights.masked_fill(self.bias[:, :, :time, :time] == 0, float('-inf'))
            weights = F.softmax(weights, dim=-1)
            weights = self.attention_dropout(weights)
            y = weights @ value # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(batch, time, channels) # re-assemble all head outputs side by side
        y = self.projection(y)
        y = self.residual_dropout(y)

        return y


class FeedForward(nn.Module):
    def __init__(self, n_embed, dropout):
        super().__init__()
        self.input = nn.Linear(n_embed,  4 * n_embed)
        self.projection = nn.Linear(4 * n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.input(x)
        x = new_gelu(x)
        x = self.projection(x)
        x = self.dropout(x)

        return x


class Block(nn.Module):
    def __init__(self, n_embed, n_heads, block_size, dropout):
        super().__init__()

        self.self_attention_heads = MultiHeadAttention(
            n_heads=n_heads,
            n_embed=n_embed,
            block_size=block_size,
            dropout=dropout
        )

        self.ffwd = FeedForward(n_embed, dropout)

        self.layer_norm_1 = nn.LayerNorm(n_embed)
        self.layer_norm_2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.self_attention_heads(self.layer_norm_1(x))
        x = x + self.ffwd(self.layer_norm_2(x))

        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, head_size, block_size, n_heads, n_blocks, dropout):
        super().__init__()
        self.block_size = block_size
        self.n_embed = head_size * n_heads
        # table mapping token to next token
        self.token_embedding_table = nn.Embedding(num_embeddings=vocab_size, embedding_dim=self.n_embed)  # B, T, C
        self.position_embedding_table = nn.Embedding(num_embeddings=block_size, embedding_dim=self.n_embed)  # T, C
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.Sequential(
            *[Block(self.n_embed, n_heads, block_size, dropout) for _ in range(n_blocks)]
        )
        self.final_layer_norm = nn.LayerNorm(self.n_embed)
        self.lm_head = nn.Linear(in_features=self.n_embed, out_features=vocab_size)  # B, T, vocab_size

        self.init_weights(n_blocks)

        self.device = torch.device("cpu")

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.position_embedding_table.weight.numel()
        return n_params

    @staticmethod
    def _init_weights_callback(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def init_weights(self, n_blocks):
        # Tie weights between token embedding and lm_head https://paperswithcode.com/method/weight-tying
        self.token_embedding_table.weight = self.lm_head.weight

        self.apply(self._init_weights_callback)

        # apply special scaled init to the residual projections, per GPT-2 paper
        for name, par in self.named_parameters():
            if name.endswith('projection.weight'):
                torch.nn.init.normal_(par, mean=0.0, std=0.02/math.sqrt(2 * n_blocks))

    def forward(self, idx, targets=None):
        batch, time = idx.size()

        assert time <= self.block_size, f"Cannot forward sequence of length {time}, block size is only {self.block_size}"

        token_embedding = self.token_embedding_table(idx)
        pos_embedding = self.position_embedding_table(
            torch.arange(0, time, dtype=torch.long, device=self.device).unsqueeze(0)
        )

        x = token_embedding + pos_embedding  # Add positional embedding
        x = self.dropout(x)
        x = self.blocks(x)
        x = self.final_layer_norm(x)

        if targets is None:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        else:
            logits = self.lm_head(x)
            batch, time, channel = logits.shape

            logits = logits.view(batch * time, channel)

            targets = targets.view(batch * time)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        # idx is a (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cropped = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # Get predictions
            logits, loss = self(idx_cropped)
            # Focus on last time step
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
