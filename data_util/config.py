import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = '/'.join(dir_path.split('/')[:-1])
sys.path.append(dir_path)

root_dir = os.path.expanduser("~")
print(root_dir)

#train_data_path = os.path.join(root_dir, "Downloads/vinbdi/data/finished_files/train.bin")
train_data_path = os.path.join(root_dir, "Downloads/vinbdi/data/finished_files/chunked/train_*")
eval_data_path = os.path.join(root_dir, "Downloads/vinbdi/data/finished_files/val.bin")
decode_data_path = os.path.join(root_dir, "Downloads/vinbdi/data/finished_files/test.bin")
vocab_path = os.path.join(root_dir, "Downloads/vinbdi/data/finished_files/vocab")
log_root = os.path.join(root_dir, "Downloads/vinbdi/data/log")
summary_path = os.path.join(root_dir, "Downloads/vinbdi/summary_dir_incar_2")
model_dir = os.path.join(summary_path, "saved_model")

# Hyperparameters
hidden_dim= 256
emb_dim= 300
batch_size= 8
max_enc_steps=400
max_dec_steps=100
beam_size=4
min_dec_steps=35
vocab_size=50000

lr=0.15
adagrad_init_acc=0.1
rand_unif_init_mag=0.02
trunc_norm_init_std=1e-4
max_grad_norm=2.0

pointer_gen = True
is_coverage = False
cov_loss_wt = 1.0

eps = 1e-12
max_iterations = 500000

use_gpu=True

lr_coverage=0.15
