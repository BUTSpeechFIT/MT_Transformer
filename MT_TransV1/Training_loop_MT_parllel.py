#!/usr/bin/python

import sys
import os
import torch
#----------------------------------------
#=========================================================
def forword_and_update(smp_no, trainflag, model, optimizer, smp_Src_labels, smp_Tgt_labels, accm_grad, clip_grad_norm):
        Decoder_out_dict = model(smp_Src_labels,smp_Tgt_labels)
        cost, CER = model.decoder.cal_performance(pred, gold,model.decoder.IGNORE_ID,normalize_length=False,smoothing=model.decoder.label_smoothing)
        
        #breakpoint()
        #output_dict={'cost':cost, 'CER':CER, 'smp_pred':pred,'smp_gold':gold}       
        output_dict = {'cost':cost, 'dec_slf_attn_list':None, 'dec_enc_attn_list':None, 'Char_cer':CER, 'Word_cer':CER}

        cost=Decoder_out_dict.get('cost')
        if trainflag:
                cost=cost/accm_grad
                cost.backward()

                if clip_grad_norm != 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                cost.detach()

                ###gradient accumilation
                if(smp_no%accm_grad)==0:
                        optimizer.step()
                        optimizer.zero_grad()

        cost_cpu = cost.item()
        return Decoder_out_dict,cost_cpu
#=========================================================


#---------------------------------------
def train_val_model(**kwargs):
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
        smp_Src_labels = torch.LongTensor(smp_Src_labels)
        smp_Tgt_labels = torch.LongTensor(smp_Tgt_labels)

        #-----------------------------------------------------------------

        smp_Src_labels = smp_Src_labels.cuda() if args.gpu else smp_Src_labels
        smp_Tgt_labels = smp_Tgt_labels.cuda() if args.gpu else smp_Tgt_labels
        #--------------------------------

        OOM=False
        if trainflag:
            try:
                Decoder_out_dict, cost_cpu = forword_and_update(smp_no, trainflag, model, optimizer, smp_Src_labels, smp_Tgt_labels, args.accm_grad, args.clip_grad_norm)

            except Exception as e:
                    if 'CUDA out of memory' in str(e):
                      OOM=True
                      torch.cuda.empty_cache()
                      print("The model in OOM condition","smp_no", smp_no, "batch size for the batch is:", smp_Src_labels.shape)
                      #break;
                    else:
                        ####print if some other error occurs
                        print("There is some other error",str(e))


            ###When there is oom eror make the batch size 2
            if OOM:
                  batch_size = smp_Src_labels.shape[0]
                  smp_Src_labels = smp_Src_labels[:2]
                  smp_Tgt_labels = smp_Tgt_labels[:2]
                  print("The model running under OOM condition", "smp_no", smp_no, "batch size for the batch is:", 2)
                  Decoder_out_dict, cost_cpu = forword_and_update(smp_no, trainflag, model, optimizer, smp_Src_labels, smp_Tgt_labels, args.accm_grad, args.clip_grad_norm)


        else:
            with torch.no_grad():
                     Decoder_out_dict, cost_cpu = forword_and_update(smp_no, trainflag, model, optimizer, smp_Src_labels, smp_Tgt_labels, args.accm_grad, args.clip_grad_norm)
        #--------------------------------

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
