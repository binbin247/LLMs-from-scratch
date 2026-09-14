# rewrite the GPT model code
import torch
import torch.nn as nn
import tiktoken
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class GPTConfig:
    # 124 M GPT model configuration parameters
    vocab_size: int = 50257
    context_length: int = 1024
    emb_dim: int = 768
    n_heads: int = 12
    n_blocks: int = 12
    drop_rate: float = 0.1
    is_qkv_bias: bool = False


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.emb_dim % cfg.n_heads == 0, "Embedding dimension must be divisible by number of heads"

        # model parameters
        self.context_length = cfg.context_length
        self.emb_dim = cfg.emb_dim
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.emb_dim // cfg.n_heads
        self.is_qkv_bias = cfg.is_qkv_bias
        self.drop_rate = cfg.drop_rate

        # define layer normalization
        self.ln_input_data = nn.LayerNorm(self.emb_dim)
        self.ln_ffn = nn.LayerNorm(self.emb_dim)

        # define components for multi-head attention
        self.q_proj = nn.Linear(self.emb_dim, self.emb_dim, bias=self.is_qkv_bias)
        self.k_proj = nn.Linear(self.emb_dim, self.emb_dim, bias=self.is_qkv_bias)
        self.v_proj = nn.Linear(self.emb_dim, self.emb_dim, bias=self.is_qkv_bias)
        self.out_proj = nn.Linear(self.emb_dim, self.emb_dim, bias = True)
        self.dropout = nn.Dropout(self.drop_rate)

        # define components for feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(self.emb_dim, 4 * self.emb_dim),
            nn.GELU(),
            nn.Linear(4 * self.emb_dim, self.emb_dim)
        )

    def forward(self, x:torch.Tensor)->torch.Tensor:
        # prepare residual connection
        shortcut = x  # (batch_size, seq_length, emb_dim)
        batch_size, seq_length, emb_dim = x.shape
        assert seq_length <= self.context_length,(
            f"Input sequence length {seq_length} exceeds context length {self.context_length}"
        )

        # layer normalization
        x = self.ln_input_data(x)  # pre-norm

        # q,k,v.shape = (batch_size, n_heads, seq_length, head_dim)
        q = self.q_proj(x).view(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)  # (batch_size, n_heads, seq_length, head_dim)
        k = self.k_proj(x).view(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)  # (batch_size, n_heads, seq_length, head_dim)
        v = self.v_proj(x).view(batch_size, seq_length, self.n_heads, self.head_dim).transpose(1, 2)  # (batch_size, n_heads, seq_length, head_dim)

        context_vec = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True, dropout_p=self.drop_rate if self.training else 0.0
        )

        context_vec = (
            context_vec
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_length, self.emb_dim)  # (batch_size, seq_length, emb_dim)
        )

        # residual connection
        context_vec = self.out_proj(context_vec)
        x = shortcut + self.dropout(context_vec)  # (batch_size, seq_length, emb_dim)

        # layer normalization
        shortcut = x  # (batch_size, seq_length, emb_dim)
        x = self.ln_ffn(x)  # (batch_size, seq_length, emb_dim)
        x = self.ffn(x)
        x = shortcut + self.dropout(x)  # (batch_size, seq_length, emb_dim)

        return x

class GPTmodel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # embedding
        self.token_emb = nn.Embedding(
            self.cfg.vocab_size, self.cfg.emb_dim
            )
        self.pos_emb = nn.Embedding(
            self.cfg.context_length, self.cfg.emb_dim
            )
        self.drop_emb = nn.Dropout(self.cfg.drop_rate)
        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(self.cfg) for _ in range(self.cfg.n_blocks)]
        )
        self.ln_output = nn.LayerNorm(self.cfg.emb_dim)
        self.out_head = nn.Linear(self.cfg.emb_dim, self.cfg.vocab_size, bias=False)


    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx.shape = (batch_size, seq_length)
        if idx.ndim != 2:
            raise ValueError("idx must have shape (batch_size, seq_length)")
        batch_size, seq_length = idx.shape
        if not 1 <= seq_length <= self.cfg.context_length:
            raise ValueError(
                f"Input sequence length {seq_length} must be between 1 and {self.cfg.context_length}"
            )
        positions = torch.arange(seq_length, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(positions)  # (batch_size, seq_length, emb_dim)
        x = self.drop_emb(x)
        x = self.transformer_blocks(x)  # (batch_size, seq_length, emb_dim)
        x = self.ln_output(x)
        logits = self.out_head(x)

        return logits  # (batch_size, seq_length, vocab_size)

def generate(model, idx: torch.Tensor, max_new_tokens: int, cfg: GPTConfig) -> torch.Tensor:
    # idx.shape = (batch_size, seq_length); call model.eval() before generation.
    for _ in range(max_new_tokens):
        # crop idx to the last context_length tokens
        idx_cond = idx[:, -cfg.context_length:]  # (batch_size, context_length)
        with torch.no_grad():
            logits = model(idx_cond)  # (batch_size, context_length, vocab_size)
        logits = logits[:, -1, :]  # (batch_size, vocab_size)
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, seq_length + 1)
    return idx

if __name__ == "__main__":
    # test generate
    cfg = GPTConfig()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = GPTmodel(cfg=cfg).to(device)
    model.eval()

    start_context = "Hell0, "
    tokenizer = tiktoken.get_encoding("gpt2")
    idx = torch.tensor(
        tokenizer.encode(start_context), dtype=torch.long, device=device
    ).unsqueeze(0)

    output = generate(model, idx, max_new_tokens=10, cfg=cfg)
    print(tokenizer.decode(output.squeeze(0).tolist()))
