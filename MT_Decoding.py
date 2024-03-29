#!/usr/bin/python
import sys
import os
from os.path import join, isdir, isfile
#----------------------------------------
import glob
import json
from argparse import Namespace


#**********
sys.path.insert(0,'/mnt/matylda3/vydana/HOW2_EXP/MT_Transformer/MT_TransV1')
#from Initializing_model_LSTM_SS_v2_args import Initialize_Att_model
from Load_sp_model import Load_sp_models
from utils__ import plotting,read_as_list
from user_defined_losses import compute_cer
from Decoding_loop import get_Bleu_for_beam
from Load_Encode_sp_model import Load_Encode_sp_model

sys.path.insert(0,'/mnt/matylda3/vydana/HOW2_EXP/MT_Transformer/MT_TransV1')
from TRANSFORMER_MT_V1 import Transformer
from Initializing_Transformer_MT import Initialize_Att_model
from Stocasting_Weight_Addition import Stocasting_Weight_Addition
#from Load_RNNLM_Model import Load_RNNLM_model
from Load_Trained_TransLm_model import Load_Transformer_LM_model

#-----------------------------------
import torch
sys.path.insert(0,'/mnt/matylda3/vydana/HOW2_EXP/MT_Transformer/MT_TransV1')
import MT_Transformer_arg
from MT_Transformer_arg import parser
args = parser.parse_args()

model_path_name=join(args.model_dir,'model_architecture_')
print(model_path_name)
#--------------------------------
###load the architecture if you have to load
with open(model_path_name, 'r') as f:
        TEMP_args = json.load(f)


ns = Namespace(**TEMP_args)
args=parser.parse_args(namespace=ns)

if not isdir(args.model_dir):
        os.makedirs(args.model_dir)



# args.Am_weight = 1
# args.LM_model = None

print(args.TransLM_model,args.RNNLM_model)
if args.Am_weight < 1:
    ##model class
    ##==================================
    #args.TransLM_model="0"
    #breakpoint()
    if (args.RNNLM_model != 'None' ) and  ( args.TransLM_model == 'None' ):
        ##AS does not need SWA so drectly give weight file 
        LM_model,_ = Load_RNNLM_model(args.RNNLM_model) 
        print("rnnlm_loop")

    elif ( args.RNNLM_model == 'None' ) and ( args.TransLM_model != 'None' ):
        print("entering to Trans_Lm loop")
        #complex as it needs SWA for the weights
        LM_model,_ = Load_Transformer_LM_model(args.TransLM_model,args.SWA_random_tag)
        print("Transformer_Lm_loop")

    else:
        print("Some thing is wrong with language models")


    #print(LM_model)
    if LM_model:
        LM_model.eval()
        LM_model = LM_model.cuda() if args.gpu else LM_model
        args.LM_model = LM_model
    else:
        args.LM_model = None


# breakpoint()



##==================================
##**********************************
##**********************************
def main():
        #Load the model from architecture
        model,optimizer=Initialize_Att_model(args)
        model.eval()
        args.gpu=False

        ###make SWA name 
        model_name = str(args.model_dir).split('/')[-1]
        ct=model_name+'_SWA_random_tag_'+str(args.SWA_random_tag)

        ##check the Weight averaged file and if the file does not exist then lcreate them
        ## if the file exists load them
        if not isfile(args.pre_trained_weight):
            if isfile(join(args.model_dir,ct)):
                model_names,checkpoint_ter = get_best_weights(args.weight_text_file,args.Res_text_file)
                model_names_checkpoints=model_names[:args.early_stopping_checkpoints]
                model = Stocasting_Weight_Addition(model,model_names_checkpoints)
                torch.save(model.state_dict(),join(args.model_dir,ct))
            else:
                print("taking the weights from",ct,join(args.model_dir,str(ct)))
                args.pre_trained_weight = join(args.model_dir,str(ct))
                model,optimizer=Initialize_Att_model(args)
        #---------------------------------------------
        model.eval() 
        #print("best_weight_file_after stocastic weight averaging")
        #=================================================
        model = model.cuda() if args.gpu else model
        plot_path=join(args.model_dir,'decoding_files','plots')
        #=================================================
        #=================================================

        ####read all the scps and make large scp with each lines as a feature
        decoding_files_list=glob.glob(args.dev_path + "*")
        scp_paths_decoding=[]
        for i_scp in decoding_files_list:
            scp_paths_decoding_temp=open(i_scp,'r').readlines()
            scp_paths_decoding+=scp_paths_decoding_temp

        #scp_paths_decoding this should contain all the scp files for decoding
        #====================================================
        ###sometime i tend to specify more jobs than maximum number of lines in that case python indexing error we get  
        job_no=int(args.Decoding_job_no)-1
        
            
        # breakpoint()
        #####get_cer_for_beam takes a list as input
        present_path = scp_paths_decoding[job_no]
        present_path = present_path.strip()
        Src_text_file_dict = {line.split(' ')[0]:" ".join(line.strip().split(' ')[1:]) for line in open(args.src_text_file)}
        Tgt_text_file_dict = {line.split(' ')[0]:" ".join(line.strip().split(' ')[1:]) for line in open(args.tgt_text_file)}
        
        key = present_path.split(' ')[0]
        Src_text = Src_text_file_dict.get(key,None)        
        Tgt_text = Tgt_text_file_dict.get(key,None)          
        
        Src_text = Src_text.strip() if Src_text else Src_text
        Tgt_text = Tgt_text.strip() if Tgt_text else Tgt_text

        print("input_file", key,Src_text)
        
        if not Src_text:
            print("utterance not present in source tokens something wrong",key)
            exit(0)
        else:
            Src_tokens = Load_Encode_sp_model(args.Src_model_path,Src_text)
        
            if Tgt_text==None:
                Tgt_tokens=None
            else:
                 Tgt_tokens = Load_Encode_sp_model(args.Tgt_model_path,Tgt_text)
        get_Bleu_for_beam(key,Src_tokens,Src_text,Tgt_tokens,Tgt_text,model,plot_path,args)

#--------------------------------
def get_best_weights(weight_text_file,Res_text_file):
        weight_list=read_as_list(weight_text_file)
        ERROR_list=read_as_list(weight_text_file+'_Res')
        weight_acc_dict = dict(zip(ERROR_list,weight_list))
       
        sorted_weight_acc_dict = sorted(weight_acc_dict.items(), key=lambda x: x[0],reverse=False)
        check_points_list = sorted_weight_acc_dict

       
        model_names=[W[1] for W in check_points_list]
        checkpoint_ter=[W[0] for W in check_points_list]
        round_checkpoint_ter=[str(round(float(N),2)) for N in checkpoint_ter]
        #print('checkpoint_TER,checkpoint_names',round_checkpoint_ter,model_names)
        return model_names,checkpoint_ter

#--------------------------------

if __name__ == '__main__':
    main()

