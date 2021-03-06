#!/usr/bin/python

import sys
import os
import torch
#----------------------------------------



#---------------------------------------
def train_val_model(**kwargs):

        #breakpoint()

        smp_no=kwargs.get('smp_no')
        args = kwargs.get('args')
        model = kwargs.get('model')
        optimizer= kwargs.get('optimizer')
 
        trainflag = kwargs.get('trainflag')
        
        B1 = kwargs.get('data_dict')
        
        smp_Src_data = B1.get('smp_Src_data')
        smp_Src_labels = B1.get('smp_Src_labels')
        smp_Tgt_labels = B1.get('smp_Tgt_labels')
        
        ###for future
        smp_Src_Text = B1.get('smp_Src_Text') 
        smp_Tgt_Text = B1.get('smp_Tgt_Text')  
         
       
        #################finished expanding the keyword arguments#########
        ##===========================================
        #============================================
        ###################################################################
        #input = torch.from_numpy(smp_Src_data).float() ####


        smp_Src_labels = torch.LongTensor(smp_Src_labels)
        smp_Tgt_labels = torch.LongTensor(smp_Tgt_labels)

        #-----------------------------------------------------------------
        #input = input.cuda() if args.gpu else input
        #breakpoint()
        smp_Src_labels = smp_Src_labels.cuda() if args.gpu else smp_Src_labels
        smp_Tgt_labels = smp_Tgt_labels.cuda() if args.gpu else smp_Tgt_labels
        #--------------------------------
        #
        OOM=False
        if trainflag:

            try:
                Decoder_out_dict = model(smp_Src_labels,smp_Tgt_labels)

            except Exception as e:
                   if 'CUDA out of memory' in str(e):
                      OOM=True
                      torch.cuda.empty_cache()
                      print("The model in OOM condition","smp_no",smp_no,"batch size for the batch is:",smp_Src_labels.shape)
                      #break;
            ###When there is oom eror make the batch size 2
            if OOM:
                  batch_size = smp_Src_labels.shape[0]
                  smp_Src_labels = smp_Src_labels[:2]
                  smp_Tgt_labels = smp_Tgt_labels[:2]
                  print("The model running under OOM condition","smp_no",smp_no,"batch size for the batch is:",2)
                  Decoder_out_dict = model(smp_Src_labels,smp_Tgt_labels)

        else:
            with torch.no_grad():
                    Decoder_out_dict = model(smp_Src_labels,smp_Tgt_labels)

        #--------------------------------
        cost=Decoder_out_dict.get('cost')

        ###training with accumilating gradients
        if trainflag:
                cost=cost/args.accm_grad
                cost.backward()
                if args.clip_grad_norm != 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip_grad_norm)

                cost.detach()   
                ###gradient accumilation
                if(smp_no%args.accm_grad)==0:
                    optimizer.step()
                    optimizer.zero_grad()
                cost_cpu=cost.item()
        #--------------------------------------
        cost_cpu = cost.item() 

        ###output a dict
        #==================================================    
        Output_trainval_dict={
                            'cost_cpu':cost_cpu,
                            'dec_slf_attn_list':Decoder_out_dict.get('dec_slf_attn_list'),
                            'dec_enc_attn_list':Decoder_out_dict.get('dec_enc_attn_list'),
                            'Char_cer':Decoder_out_dict.get('Char_cer'),
                            'Word_cer':Decoder_out_dict.get('Word_cer')}
        return Output_trainval_dict
#=========================================================
