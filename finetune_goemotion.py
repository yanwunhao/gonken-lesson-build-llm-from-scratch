"""
Fine-tune GPT-2 on the GoEmotions dataset.
GoEmotions is a Reddit comments dataset with emotion labels.

Before running this, prepare the data:
    python data/goemotion/prepare.py

Then run fine-tuning:
    python train.py config/finetune_goemotion.py

After training, generate samples:
    python sample.py --out_dir=out-goemotion \
    --start="Emotion: joy\nText:" \
    --max_new_tokens=50 \
    --num_samples=1
"""

import time

# Output directory
out_dir = 'out-goemotion'

# Evaluation settings
eval_interval = 10
eval_iters = 10
log_interval = 10

# Logging
wandb_log = False  # feel free to turn off
wandb_project = 'goemotion'
wandb_run_name = 'ft-gpt2-' + str(time.time())

# Dataset
dataset = 'goemotion'

# Initialize from pre-trained GPT-2
# Options: 'gpt2' (124M), 'gpt2-medium' (350M), 'gpt2-large' (774M), 'gpt2-xl' (1558M)
init_from = 'gpt2'  # start with base GPT-2, can upgrade to larger models

# Checkpointing
always_save_checkpoint = False  # only save when validation loss improves

# Training settings
# GoEmotions train set has ~43k examples, ~6.5M tokens (estimated)
# With these settings: 4 batch_size * 16 grad_accum * 1024 tokens = 65,536 tokens/iter
# So 1 epoch ~= 100 iterations
batch_size = 4
gradient_accumulation_steps = 16
block_size = 1024

# Training iterations
# Training for ~1 epochs
max_iters = 100

# Learning rate settings
# Use constant LR for fine-tuning (common practice)
learning_rate = 3e-5  # lower learning rate for fine-tuning
decay_lr = False

# Alternatively, you can use learning rate decay:
# decay_lr = True
# warmup_iters = 100
# lr_decay_iters = 1000
# min_lr = 3e-6

# Regularization
dropout = 0.1  # add some dropout for regularization during fine-tuning
weight_decay = 0.01

# System
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True  # use PyTorch 2.0 compile for faster training
