#! /usr/bin/python

#*******************************
import sys
import os
from os.path import join, isdir
from random import shuffle
import glob

import random
import string

sys.path.insert(0,'/mnt/matylda3/vydana/HOW2_EXP/MT_Transformer')
from MT_TransV1.Load_sp_model import Load_sp_models
from MT_TransV1.Make_ASR_scp_text_format_fast import format_tokenize_data

import MT_TransV1.MT_Transformer_arg
from MT_TransV1.MT_Transformer_arg import parser
args = parser.parse_args()


if not isdir(args.data_dir):
        os.makedirs(args.data_dir)


letters=string.ascii_uppercase + string.digits
scp_file_name=(''.join(random.choice(letters) for i in range(25)))
scp_file=open(join(os.getcwd(),'scp_temp',scp_file_name),'w')
for line in open(args.tgt_text_file,'r'):
	uttid=line.split(" ")[0]
	print(uttid,file=scp_file)
scp_file.close()


final_scp_name=str(args.tgt_text_file).split('/')[-1]
train_scp_file=open(join(args.data_dir,final_scp_name + '__train_scp'),'w')

# print(args.src_text_file,args.tgt_text_file)
# exit(0)
#

format_tokenize_data(scp_files=glob.glob(join(os.getcwd(),'scp_temp',scp_file_name)) ,transcript=args.src_text_file,Translation=args.tgt_text_file,outfile=train_scp_file,Src_model_path=args.Src_model_path,Tgt_model_path=args.Tgt_model_path)

format_tokenize_data(scp_files=glob.glob(args.dev_path + "*"),transcript=args.src_text_file,Translation=args.tgt_text_file,outfile=open(join(args.data_dir,'dev_scp'),'w'), Src_model_path=args.Src_model_path,Tgt_model_path=args.Tgt_model_path)


