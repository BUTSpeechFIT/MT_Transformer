#1.Initially train tokenizers and they lead the further experiments and contol the vocabulary and other tags
        1.Source Tokenizers, Target Tokenizers

#Run.sh
Stage-1. Tokenizers are used to do data-prepearation and you can see it in MT_data_files
                                                                                ------->MT_data_files/train_scp
                                                                                ------->MT_data_files/dev_scp

                                                                                ###             uttid @@@ scp_path @@@ None @@@ Src_text @@@ Src_tokens @@@ Tgt_text @@@ Tgt_tokens @@@
                                                                                ##Data format: --8pSDeC-fg_0 @@@@ None @@@@ None @@@@ Hi. @@@@ 563 12 @@@@ Oi. @@@@ 737 12 @@@@



## 1.MT_training.py --------training the model has 
        
###code         
        epoch loop:
                training loop:
                validation loop:
                
                End of epoch stuff:
                        switch on/off flags 
                        early stopping


## 2.Training_loop_Mt.py
        read data_dict from dataloader and does forward_prop, and Weight updation
        
## 3.Initializing_Transformer_MT.py
        Initialize the model model
        if pretrained_weight !="0":
                load the pretrained_weight to the model

##4.Transformer model
        TRANSFORMER_MT_V1.py

##5.MT_Decoding.py
        1. Check the weight_text_file and Res_text_file sort and pick the n-best checkpoints and average them and save as 
                                                                                                                        SWA_weights=model_name+'_SWA_random_tag_'+str(args.SWA_random_tag)
        
        2. if running as multiple threads in parllel threads attempt to create SWA_weights i run a loop twice 
                                                                        
                                                                        for max_parlle_jobs in 1, max_parlle_jobs:
                                                                                
                                                                                if not SWA_weights.exists():
                                                                                        SWA_Average()                                                                                
         
        3. queue.pl passes the job_no and reads the line in 

                        Eng_text_line = dev_path/utt_list[job_no]
        
        get_Bleu_for_beam(Eng_text_line,......)

                4. input_data=Src_model.EncodeASids(Eng_text_line)  

                5. output_dict = model.predict(input_data)

                6. hyp_Tgt_text_seq = output_dict['Text_seq']
       
                7. Bleu_score(hyp_Tgt_text_seq,ref_line) ####not correct just for reference

##Runs in bash 
6. Ref_file = log_path/scoring/ref_file
7. Hyp_file = log_path/scoring/hyp_file



##compute corpus Bleu and this is the corect value 
python $scoring_path/compute_scare_corpus_Blue.py Ref_file Hyp_file









 
