# demo conversion
import os 
import torch
import pickle
import numpy as np
from utils import pad_seq_to_2
from utils import quantize_f0_numpy
from model import Generator_3 as Generator
from model import Generator_6 as F0_Converter

from utils import pad_seq_to_2, quantize_f0_torch, quantize_f0_numpy
from model import InterpLnr

import argparse
from torch.backends import cudnn

from solver import Solver
from data_loader import get_loader
from hparams import hparams, hparams_debug_string

min_len_seq = hparams.min_len_seq
max_len_seq = hparams.max_len_seq
max_len_pad = hparams.max_len_pad

root_dir = 'assets/spmel'
feat_dir = 'assets/raptf0'

torch.set_default_dtype(torch.float32)
device = 'cuda:0'
Interp = InterpLnr(hparams)

G = Generator(hparams).eval().to(device)
g_checkpoint = torch.load('assets/45000-G.ckpt', map_location=lambda storage, loc: storage)
G.load_state_dict(g_checkpoint['model'])

P = F0_Converter(hparams).eval().to(device)
p_checkpoint = torch.load('assets/640000-P.ckpt', map_location=lambda storage, loc: storage)
P.load_state_dict(p_checkpoint['model'])

### Load dataset
# For fast training.
cudnn.benchmark = True

torch.set_default_dtype(torch.float32)
# Data loader.
vcc_loader = get_loader(hparams)


############################3
# Pick First Voice For now (Todo: choose?)

data_loader_samp = vcc_loader[2]
data_iter_samp = iter(data_loader_samp)
speaker_id_name, x_real_pad, emb_org_val, f0_org_val, len_org_val = next(data_iter_samp)

x_real_pad = x_real_pad.to(device)
emb_org_val = emb_org_val.to(device)
len_org_val = len_org_val.to(device)
f0_org_val = f0_org_val.to(device)

x_f0 = torch.cat((x_real_pad, f0_org_val), dim=-1)
x_f0_F = torch.cat((x_real_pad, torch.zeros_like(f0_org_val)), dim=-1)
x_f0_C = torch.cat((torch.zeros_like(x_real_pad), f0_org_val), dim=-1)

x_f0_intrp = Interp(x_f0, len_org_val) 
f0_org_intrp = quantize_f0_torch(x_f0_intrp[:,:,-1])[0]
x_f0_intrp_org = torch.cat((x_f0_intrp[:,:,:-1], f0_org_intrp), dim=-1)

x_f0_F_intrp = Interp(x_f0_F, len_org_val) 
f0_F_org_intrp = quantize_f0_torch(x_f0_F_intrp[:,:,-1])[0]
x_f0_F_intrp_org = torch.cat((x_f0_F_intrp[:,:,:-1], f0_F_org_intrp), dim=-1)

x_f0_C_intrp = Interp(x_f0_C, len_org_val) 
f0_C_org_intrp = quantize_f0_torch(x_f0_C_intrp[:,:,-1])[0]
x_f0_C_intrp_org = torch.cat((x_f0_C_intrp[:,:,:-1], f0_C_org_intrp), dim=-1)
                        
x_identic_val = G(x_f0_intrp_org, x_real_pad, emb_org_val)
x_identic_woF = G(x_f0_F_intrp_org, x_real_pad, emb_org_val)
x_identic_woR = G(x_f0_intrp_org, torch.zeros_like(x_real_pad), emb_org_val)
x_identic_woC = G(x_f0_C_intrp_org, x_real_pad, emb_org_val)

conditions = ['N', 'F', 'R', 'C']
spect_vc = []
with torch.no_grad():
    for condition in conditions:
        if condition == 'N':
            x_identic_val = G(x_f0_intrp_org, x_real_pad, emb_org_val)
        if condition == 'F':
            x_identic_val = G(x_f0_F_intrp_org, x_real_pad, emb_org_val)
        if condition == 'R':
            x_identic_val = G(x_f0_intrp_org, torch.zeros_like(x_real_pad), emb_org_val)
        if condition == 'C':
            x_identic_val = G(x_f0_C_intrp_org, x_real_pad, emb_org_val)
            
        uttr_trg = x_identic_val[0].cpu().numpy().T
                
        spect_vc.append( (speaker_id_name, uttr_trg ) )  