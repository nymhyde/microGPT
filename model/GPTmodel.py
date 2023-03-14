#! /usr/bin/env python3

# GPT Language Model #
# ------------------ #

import math, inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F



@torch.jit.script
def gelu(x):
    """
    - Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    - Implementation : GELU Activation Function from Google BERT
    """

    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class SelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projection
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # requires PyTorch 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')



    def forward(self, x):
        B, T, C = x.size()      # batch size, sequence length, embedding dims (n_embd)

        # calculate query, key, values for all heads in batch
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)    # q, k, v in (B, T, C) shape
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)   # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)   # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)   # (B, nh, T, hs)

        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                               dropout_p = self.dropout,
                                               is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs

        y = self.resid_dropout(self.c_proj(y))

        return y



class MLP(nn.Module):
    '''
    Fully connected layer with GELU activation
    '''

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)


    def forward(self, x):
        x = gelu(self.c_fc(x))
        x = self.c_proj(x)
        x = self.dropout(x)

        return x


class Block(nn.Module):
    '''
    Combining Layer Normalization + SelfAttention + FullyConnected Layers
    '''

    def __init__(self, config):
        super().__init__()

        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = SelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        return x



@dataclass
class GPTConfig:
    block_size : int = 1024
    vocab_size : int = 50257
    n_layer    : int = 12
    n_head     : int = 12
    n_embd     : int = 768

    dropout    : float = 0.0
    bias       : bool = True


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None

        self.config = config


        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias = config.bias),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # init weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projection, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)



    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()

        # t cannot be longer than block_size
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size {self.config.block_size}"


        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)


        if targets is not None:
            logits = self.lm_head(x)
            loss   = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        else:
            logits = self.lm_head(x[:, [-1], :])    # gives the last time / context window
            loss = None

        return logits, loss


    def get_model(self):
        # no pretrained model loading
        # microGPT from scratch initialized
        config = GPTConfig()
        model  = GPT(config)

        return model


    def config_optimizer(self, weight_decay, lr, betas, device_type):
        """
        Seperating out all parameters into two buckets ::
            - one that will experience weight decay
            - others that won't (biases, layernorm/embedding weights)
        """

        decay = set()
        no_decay = set()
        whitelist_weight = (torch.nn.Linear, )
        blacklist_weight = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn      # full param name
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight):
                    no_decay.add(fpn)

        decay.remove('lm_head.weight')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay

        assert len(inter_params) == 0, f"parameters {str(inter_params)} are common to both group"
        assert len(param_dict.keys() - union_params) == 0, f"parameters {str(param_dict.keys() - union_params)} were not seperated"

        # creating the pytorch optimizer
        optim_groups = [
                {"params" : [param_dict[pn] for pn in sorted(list(decay))],
                 "weight_decay" : weight_decay},
                {"params" : [param_dict[pn] for pn in sorted(list(no_decay))],
                 "weight_decay" : 0.0},
                       ]

        # checking the PyTorch nightly new 'fused' option for AdamW
        use_fused = (device_type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
        print(f'Using fused AdamW : {use_fused}')

        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas, **extra_args)


        return optimizer



    # generateing text
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temp=1.0):
        """
        - Takes a conditioning sequence of indices idx (shape (b,t)) and complete the
        seq max_new_tokens times.
        - model.eval() --> no weight update. so back grad is not required.
        """

        for _ in range(max_new_tokens):
            # if sequence context is too long, we need to crop it at block size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward pass
            logits, _ = self(idx_cond)
            # logits from last step and scaling by temperature
            logits = logits[:, -1, :] / temp

            # softmax
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # appending sampled index to the running seq and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
