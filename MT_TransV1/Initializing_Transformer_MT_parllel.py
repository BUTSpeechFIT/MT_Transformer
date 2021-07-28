#!/usr/bin/python
import sys
import os
from os.path import join, isdir
import torch
from torch import optim,nn

sys.path.insert(0,'/mnt/matylda3/vydana/HOW2_EXP/MT_Transformer')
from MT_TransV1.TRANSFORMER_MT_V1 import Transformer,TransformerOptimizer
from MT_TransV1.utils__ import count_parameters
#====================================================================================
def Initialize_Att_model(args):
        
        model = Transformer(args)
        ###Dataparllel



        trainable_parameters=list(model.parameters())    
        ###optimizer
        optimizer = optim.Adam(params=trainable_parameters,lr=args.learning_rate,betas=(0.9, 0.99))

        #optimizer with warmup_steps
        optimizer=TransformerOptimizer(optimizer=optimizer,k=args.lr_scale,step_num=args.step_num, d_model=args.decoder_dmodel, warmup_steps=args.warmup_steps)

        pre_trained_weight=args.pre_trained_weight
        weight_flag=pre_trained_weight.split('/')[-1]
        print("Initial Weights",weight_flag)

        if weight_flag != '0':
                print("Loading the model with the weights form:",pre_trained_weight)
                weight_file=pre_trained_weight.split('/')[-1]
                weight_path="/".join(pre_trained_weight.split('/')[:-1])
                enc_weight=join(weight_path,weight_file)

                #-------------------------------------------------------------------------------
                try:
                        enc_weight_state_dict=torch.load(enc_weight, map_location=lambda storage, loc: storage)
                        if enc_weight_state_dict.get('module.decoder.layer_stack.0.pos_ffn.w_1.weight') != None:
                                        print('with data parllel')
                                        model=nn.DataParallel(model)                                        
                                        model.load_state_dict(torch.load(enc_weight, map_location=lambda storage, loc: storage),strict=True)

                        elif enc_weight_state_dict.get('decoder.layer_stack.0.pos_ffn.w_1.weight') != None:
                                    model.load_state_dict(torch.load(enc_weight, map_location=lambda storage, loc: storage), strict=True)                        
                                    #model.load_state_dict(torch.load(enc_weight, map_location=lambda storage, loc: storage),strict=True)                
                ##-------------------------------------------------------------------------------

                except Exception as e:
                   print(e)
                   if 'RuntimeError' in str(e):
                        model.load_state_dict(torch.load(enc_weight, map_location=lambda storage, loc: storage),strict=False)                


        model= model.cuda() if args.gpu else model
        print("model:=====>",(count_parameters(model))/1000000.0)
        return model, optimizer
#====================================================================================
#====================================================================================
