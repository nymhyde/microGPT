"""
Trainer File ::

- Single GPU run
$ python3 train.py --batch_size=32
"""

import os, time, math
import pickle               # serializing object structure

import torch
import numpy as np

# import the defined model
from model.GPTmodel import GPTConfig, GPT


# ----------------------------------------
#       << Default Config Values >>
# ----------------------------------------



# i/o
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'

# wandb logging
wandb_log = False
wandb_project = 'gpt'
wandb_run_name = 'gpt2'

# data
dataset = 'openwebtext'

# model
batch_size = 32
block_size = 1024
vocab_size = 50257  ## 50304 for efficiency ?

n_layer = 12
n_head = 12
n_embd = 768

dropout = 0.0
bias = False

# adam optim
lr = 6e-4
max_iters = 500000     # 5,00,000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95

# lr decay
decay_lr = True
warmup_iters = 2000
lr_decay_iters = max_iters
min_lr = 6e-5

# device
device = 'cuda'
dtype = 'float16'

# i/o setup
os.makedirs(out_dir, exist_ok = True)

# seed
torch.manual_seed(44)

# ----------------------------------------

# setup autocast
ptdtype = {'float32' : torch.float32, 'bfloat16' : torch.bfloat16,
           'float16' : torch.float16}[dtype]
ctx = torch.amp.autocast(device_type = device, dtype=ptdtype)



# data loader
data_dir = os.path.join('data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype = np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype = np.uint16, mode='r')

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

    if device == 'cuda':
        # pin array x, y which allows us to move them to GPU async (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)

    return x, y

# model and other inits
iter_num = 0
best_val_loss = 1e9

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)

# Init a new model from scratch
if init_from == 'scratch':

    print(f"*** Initializing a new model from scratch ***")
    print(f"\n\nDefaulting to vocab_size to {vocab_size}")

    model_args['vocab_size'] = vocab_size
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

    # to device
    model.to(device)


# initialize a Gradient
'''
-- solves the problem of underflowing
---- float16 tensors often don't take exteremly small variations in gradient
---- to prevent this, we scale our gradients by some factor
---- ** this is not vanishing gradient **
'''
scaler = torch.cuda.amp.GradScaler(enabled=(dtype=='float16'))


# optimizer
optimizer = model.config_optimizer(weight_decay, lr, (beta1, beta2), device)

# compile the model
'''
-- Speed up the PyTorch code
---- makes code run faster by JIT-compiling PyTorch -> optimized kernels
'''
if compile:
    print(f"Compiling the model ... ")
    uncompiled_model = model
    model = torch.compile(model)        # PyTorch 2.0


# helps estimate an arbitrarily acc loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()

    return out


# logging
if wandb_log :
    import wandb
    wandb.init(project = wandb_project, name=wandb_run_name, config=config)

# ---------------
# training loop
# ---------------

X, Y = get_batch('train')           # only fetched the first batch
t0 = time.time()
local_iter = 0
raw_model = model

while True:
    # setting the lr
    for param_group in optimizer.param_groups:
        param_group['lr'] =lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter_num} :: train loss {losses['train']:.4f} :: val loss {losses['val']:.4f}")

        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {'model' : raw_model.state_dict(),
                              'optimizer' : optimizer.state_dict(),
                              'model_args' : model_args,
                              'iter_num' : iter_num,
                              'best_val_loss' : best_val_loss,
                              'config' : config,
                              }

                print(f"Saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

    if iter_num == 0 and eval_only:
        break

    with ctx:
        logits, loss = model(X,Y)

    # forward pass
    X,Y = get_batch('train')
    # backward pass
    scaler.scale(loss).backward()
    # step the optimizer
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients asap
    optimizer.zero_grad(set_to_none=True)


    # timinig and logging
    t1 = time.time()
    d1 = t1 - t0
    t0 = t1

    if iter_num % log_interval == 0:
        lossf = loss.item()

        print(f"iter {iter_num} :: loss {lossf:.4f} :: time{dt*1000:.2f}ms")

    iter_num += 1
    local_iter += 1

    # termination conditions
    if iter_num > max_iters:
        break
