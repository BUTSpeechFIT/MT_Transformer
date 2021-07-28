#!/usr/bin/python
import sys
import os
import subprocess
from os.path import join, isdir
import numpy as np
import fileinput
import json
import random
from itertools import chain
from numpy.random import permutation
##------------------------------------------------------------------
import torch
from torch.autograd import Variable
#----------------------------------------
import torch.nn as nn
from torch import autograd, nn, optim
os.environ['PYTHONUNBUFFERED'] = '0'
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim

from random import shuffle
from statistics import mean
import matplotlib
import matplotlib.pyplot as plt 
plt.switch_backend('agg')
matplotlib.pyplot.viridis()
import glob

#*************************************************************************************************************************
####### Loading the Parser and default arguments
sys.path.insert(0,'/mnt/matylda3/vydana/HOW2_EXP/MT_Transformer/MT_TransV1/')
import MT_Transformer_arg
from MT_Transformer_arg import parser
args = parser.parse_args()

###save architecture for decoding
model_path_name=join(args.model_dir,'model_architecture_')
with open(model_path_name, 'w') as f:
    json.dump(args.__dict__, f, indent=2)
print(args)
# #####setting the gpus in the gpu cluster
# #**********************************
from Set_gpus import Set_gpu
if args.gpu:
    Set_gpu()
###----------------------------------------
#==============================================================
from Dataloader_for_MT_v2_check import DataLoader
from TRANSFORMER_MT_V1 import Transformer
from Initializing_Transformer_MT import Initialize_Att_model
from Training_loop_MT import train_val_model
from Load_sp_model import Load_sp_models
##==================================
#==============================================================
############################################
#=============================================================
def main():
        ##Load setpiece models for Dataloaders
        Src_model=Load_sp_models(args.Src_model_path)
        Tgt_model=Load_sp_models(args.Tgt_model_path)
        ###initilize the model
        #============================================================
        #------------------------------------------------------------  
        #/mnt/matylda3/vydana/HOW2_EXP/IWSLT2021_MT_systems/Data_files_Mustc/train_splits
        #dev_splits
        #train_splits_V2
        #train_splits_V2/*
        train_gen = DataLoader({'files': glob.glob(args.data_dir + "dev_scp_sorted"),
                                'max_batch_label_len': args.max_batch_label_len,
                                'max_batch_len': args.max_batch_len,
                                'max_feat_len': args.max_feat_len,
                                'max_label_len': args.max_label_len,
                                'Src_model': Src_model,
                                'Tgt_model': Tgt_model,
                                'queue_size': 100,
                                'apply_cmvn': 1,
                                'min_words': 0,
                                'max_words': args.max_words,
                                'min_len_ratio': args.min_len_ratio})

        sentences=0        
        for i in range(166563):
                B1 = train_gen.next()
                assert B1 is not None, "None should never come out of the DataLoader"
                smp_Src_labels = B1.get('smp_Src_labels')
                smp_Tgt_labels = B1.get('smp_Tgt_labels')
                sentences+=smp_Src_labels.shape[0]
                print(':===>',i, smp_Src_labels.shape, smp_Tgt_labels.shape,"sentences:-->",sentences)
      
#=======================================================
#=============================================================================================
if __name__ == '__main__':
    main()



