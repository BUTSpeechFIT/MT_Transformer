#! /usr/bin/python

import sys
import os
from os.path import join


sys.path.insert(0,'/mnt/matylda3/vydana/HOW2_EXP/MT_Transformer/MT_TransV1')
from MT_TransV1.Load_sp_model import Load_sp_models
text_dlim=' @@@@ '
##================================================================
##================================================================
# output_file='Timit_text_like_MT'
# scp_file='/mnt/matylda3/vydana/benchmarking_datasets/Timit/scp_files/train/sorted_feats_pdnn_train_scp'
# transcript='/mnt/matylda3/vydana/benchmarking_datasets/Timit/All_text'
# Translation='/mnt/matylda3/vydana/benchmarking_datasets/Timit/All_text_2'
# Word_model_path='/mnt/matylda3/vydana/benchmarking_datasets/Timit/models/Timit_PHSEQ_100/Timit_PHSEQ__100__word.model'
# Word_model = Load_sp_models(Word_model_path)
# Char_model = Word_model
# outfile=open(output_file,'w')
#=====================================================================================
def Search_for_utt(query, search_file,SPmodel_path,model):
        #check if the model is word model
        ##this whole nonsence is due to thesentenc epiece mapping two consecutive OOVs as a single OOV this is undesirable and 
        ##other tokeniations such as bpe ,char,unigram doesnot have this .....so for the word models ame ends with __word so encode the text word by word
        #not a good hack ...but i dont know it for now 
        #---------------------
        #---------------------
        SPmodel=model
        utt_text=search_file.get(query,'None')
        utt_text=" ".join(utt_text)

        if utt_text != 'None' and SPmodel_path:
                this_is_word_model = True if '__word' in SPmodel_path else False
                ##check if the model is a word model and load it
                #SPmodel=Load_sp_models(SPmodel_path)
                
                tokens_utt_text=[]
                if this_is_word_model:
                    for word in utt_text.split(' '):
                        tokens_utt_text += SPmodel.EncodeAsIds(word)

                else:
                    tokens_utt_text = SPmodel.EncodeAsIds(utt_text)

                tokens_utt_text = [str(intg) for intg in tokens_utt_text]
                tokens_utt_text = " ".join(tokens_utt_text)
        else:

            tokens_utt_text = 'None'
        
        utt_text = utt_text + text_dlim + tokens_utt_text + text_dlim
        return utt_text

#=======================================================================================
def format_tokenize_data(scp_files,transcript,Translation,outfile,Tgt_model_path,Src_model_path): 
        for scpfile in scp_files:  
          #import pdb;pdb.set_trace() 
          
          print(scpfile)
          line=open(scpfile).readline()
          line=line.split(' ')[0]
          if len(line)==1:
            ###-------:for MT data
            scp_dict={line.strip().split(' ')[0]:None for line in open(scpfile)}
          else:
            scp_dict={line.strip().split(' ')[0]:line.strip().split(' ')[1:] for line in open(scpfile)}
          #=====================================================================
          transcript_dict={line.strip().split(' ')[0]:line.strip().split(' ')[1:] for line in open(transcript)}
          Translation_dict={line.strip().split(' ')[0]:line.strip().split(' ')[1:] for line in open(Translation)}
        
          print('done loading dicts')
          SPmodel=Load_sp_models(Src_model_path)
          Tgmodel=Load_sp_models(Tgt_model_path)
           
          for query in list(scp_dict.keys()):
                #print(query)
                inp_seq = query + text_dlim
                inp_seq += Search_for_utt(query, search_file=scp_dict,SPmodel_path=None,model=None)
                inp_seq += Search_for_utt(query, search_file=transcript_dict,SPmodel_path=Src_model_path,model=SPmodel)
                inp_seq += Search_for_utt(query, search_file=Translation_dict,SPmodel_path=Tgt_model_path,model=Tgmodel)
                #------------------
                #print(inp_seq)
                print(inp_seq,file=outfile) 
#============================================================================
#format_tokenize_data([scp_file],transcript,Translation,outfile,Word_model,Char_model)
      
















