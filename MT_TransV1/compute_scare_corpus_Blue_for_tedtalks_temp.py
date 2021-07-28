#! /usr/bin/python

import numpy as np
from  statistics import mean

import sys
import sacrebleu
from sacrebleu import sentence_bleu,corpus_bleu
SMOOTH_VALUE_DEFAULT=1e-8

import os
import sys
from os.path import isdir

hyp_file = open(sys.argv[1],'r').readlines()
ref_file = open(sys.argv[2],'r').readlines()
hyp_file_dict={i.strip().split(' ')[0].replace("-","_"):" ".join(i.strip().split(' ')[1:]) for i in hyp_file}
ref_file_dict={i.strip().split(' ')[0].replace("-","_"):" ".join(i.strip().split(' ')[1:]) for i in ref_file}
log_folder = sys.argv[3]


log_folder = os.path.join(log_folder,'hyp_talks_moses')
if not os.path.isdir(log_folder):
        os.makedirs(log_folder)

#ref_talk_list=['/mnt/matylda3/vydana/kaldi/egs/MUSTC_V2/dev_uttlist',
#'/mnt/matylda3/vydana/kaldi/egs/MUSTC_V2/tst-COMMON_uttlist', 
#'/mnt/matylda3/vydana/kaldi/egs/MUSTC_V2/tst-HE_uttlist']

#ref_talk_list=['/mnt/matylda3/vydana/espnet_latest/espnet_JAN2020/espnet/egs/must_c/mt1_mustc_V2/dev_uttlist','/mnt/matylda3/vydana/espnet_latest/espnet_JAN2020/espnet/egs/must_c/mt1_mustc_V2/tst-COMMON_uttlist','/mnt/matylda3/vydana/espnet_latest/espnet_JAN2020/espnet/egs/must_c/mt1_mustc_V2/tst-HE_uttlist']

#ref_talk_list=['/mnt/matylda3/vydana/kaldi/egs/MUSTC_V2/dev_uttlist','/mnt/matylda3/vydana/kaldi/egs/MUSTC_V2/tst-COMMON_uttlist','/mnt/matylda3/vydana/kaldi/egs/MUSTC_V2/tst-HE_uttlist']
ref_talk_list=['/mnt/matylda3/vydana/kaldi/egs/MUSTC_V2/tst-COMMON_uttlist']

for ted_talks in ref_talk_list:
        teds = open(ted_talks,'r')
        teds = teds.readlines()
        Dataset_name=ted_talks.split('/')[-1]
        Dataset_saving_file_hyp = open(os.path.join(log_folder,Dataset_name+'_Dataset_hyp_for_moses_'),'w+')
        Dataset_saving_file_ref = open(os.path.join(log_folder,Dataset_name+'_Dataset_ref_for_moses_'),'w+')

        for utt_idx in teds:
                utt_idx = utt_idx.strip()
                utt_idx = utt_idx.replace('-','_')

                hyp_text = hyp_file_dict.get(utt_idx,None)
                ref_text = ref_file_dict.get(utt_idx,None)
                #breakpoint()

                if hyp_text==None:
                    print("Hyp text is none--------------------------------------------------------------------------------------------------------************-----------------")
                    #exit(0)
                  
                if ref_text==None:
                    print("ref text is none", 'careful-----------------')

                print(utt_idx,hyp_text, file=Dataset_saving_file_hyp)
                print(utt_idx,ref_text, file=Dataset_saving_file_ref)
                    #breakpoint()               

exit(0)

#=========================================================================================
def get_input_tuple_list(inp_file):
        inp_touple_list=[]

        for line in inp_file:
                line=line.strip()
                if line:
                        utt_id=line.split(' ')[0].replace('-','_').split('_')
                        if len(line.strip().split(' ')) >1:
                                text=" ".join(line.strip().split(' ')[1:])
                        else:
                                text=line.strip().split(' ')[1:]
                        utt_id.append(text)
                        inp_touple_list.append(tuple(utt_id))
        return inp_touple_list

#=========================================================================================

def pop_from_list(present_talk_list):
        ref_text=[]
        ref_ids=[]
        while present_talk_list:
              sent_touple = present_talk_list.pop(0)
              ref_ids.append("--".join(sent_touple[:len(sent_touple)-1]))

              #ref_text = ref_text + sent_touple[-1]
              #sent_list = sent_touple[-1].split(" ") if ' ' in sent_touple[-1] else sent_touple[-1]   
              ref_text.append(sent_touple[-1])
        return ref_text, ref_ids

#=========================================================================================
#print(get_input_tuple_list(hyp_file),get_input_tuple_list(ref_file))
#print(inp_touple_list)
#=============================================
hyp_tuple_list = get_input_tuple_list(hyp_file)
ref_tuple_list = get_input_tuple_list(ref_file)
#=============================================

final_scores_pertalk={}
#no_of_files=inp_file
ref_talk_list=['/mnt/matylda3/vydana/kaldi/egs/MUSTC_V2/dev_tedtalk_list',
'/mnt/matylda3/vydana/kaldi/egs/MUSTC_V2/tst-Common_ted_talk_list', 
'/mnt/matylda3/vydana/kaldi/egs/MUSTC_V2/tst-HE_ted_talk_list']
for ted_talks in ref_talk_list:
        teds = open(ted_talks,'r')
        teds = teds.readlines()

        final_scores_pertalk={}
        All_talks_hyp_sent=[]
        All_talks_ref_sent=[]
        Dataset_name=ted_talks.split('/')[-1]

        Dataset_saving_file_hyp = open(os.path.join(log_folder,Dataset_name+'_Dataset_hyp_for_moses_'),'w+')
        Dataset_saving_file_ref = open(os.path.join(log_folder,Dataset_name+'_Dataset_ref_for_moses_'),'w+')



        for ted in teds:
                ted = ted.strip()
                ted = ted.replace('-','_')
                ted_tup = tuple(ted.split('_'))

                present_talk_list=[ i for i in hyp_tuple_list if "--".join(ted_tup)=="--".join(i[:len(ted_tup)])]
                Ref_present_talk_list=[ i for i in ref_tuple_list if "--".join(ted_tup)=="--".join(i[:len(ted_tup)])]

                hyp_text, hyp_ids = pop_from_list(present_talk_list)
                ref_text, ref_ids = pop_from_list(Ref_present_talk_list)              
                
                breakpoint()
                All_talks_hyp_sent.extend(hyp_text)
                All_talks_ref_sent.extend(ref_text)

                #####WRTE in to textfile for moses bleu
                print(hyp_ids,hyp_text, file=Dataset_saving_file_hyp)
                print(ref_ids,ref_text, file=Dataset_saving_file_ref)



                #print(ref_text, ref_ids,ref_text, ref_ids)

                Bleu_score = corpus_bleu(hyp_text,[ref_text],smooth_value=SMOOTH_VALUE_DEFAULT,smooth_method='exp',use_effective_order='True')
                print('BLUE:===>',Bleu_score.score)
                breakpoint()
                ted_name="-".join(ted_tup)

                #final_text=hyp_text+'\t'+ref_text+'\t'+str(Bleu_score.score)                
                final_text="---".join(hyp_text)+'\t'+"---".join(ref_text)+'\t'+str(Bleu_score.score)
                
                with open(os.path.join(log_folder,ted_name+'_hyp_text_'),'a+') as tedtalk_saving_file:
                        print(final_text, file=tedtalk_saving_file)




        Dataset_saving_file_hyp.close()
        Dataset_saving_file_ref.close()

        breakpoint()
        final_scores_pertalk[ted_name]=Bleu_score.score
        print(ted_talks,mean(final_scores_pertalk.values()))
        with open(os.path.join(log_folder,ted_name+'_hyp_Bleu_'),'a+') as tedtalk_saving_file:
                print(ted_talks, mean(final_scores_pertalk.values()), file=tedtalk_saving_file)

        breakpoint()
        Dataset_name=ted_talks.split('/')[-1]
        Dataset_Bleu_score = corpus_bleu(All_talks_hyp_sent,[All_talks_ref_sent],smooth_value=SMOOTH_VALUE_DEFAULT,smooth_method='exp',use_effective_order='True')
        with open(os.path.join(log_folder,Dataset_name+'_Dataset_Bleu_'),'a+') as Dataset_saving_file:
                print(Dataset_name,Dataset_Bleu_score.score, file=Dataset_saving_file)
        print(Dataset_name,Dataset_Bleu_score.score)

