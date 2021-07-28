


tgt_lang=de
compute_dtok_blue(){
ref_file=$1
hyp_file=$2


cut -d " " -f1 $ref_file > uttid
cut -d " " -f2- $ref_file > ref_utttext
cut -d " " -f2- $hyp_file > hyp_utttext

cat ref_utttext| sed -e "s/  \+/ /g" -e "s/(/ (/" -e "s/<space>/ /g" -e "s/>/> /g">ref_utttext.detok1
cat hyp_utttext| sed -e "s/  \+/ /g" -e "s/(/ (/" -e "s/<space>/ /g" -e "s/>/> /g">hyp_utttext.detok1

perl /mnt/matylda3/vydana/HOW2_EXP/MOSES_DECODER/mosesdecoder/scripts/tokenizer/detokenizer.perl -l ${tgt_lang} -q < ref_utttext.detok1 > ref_utttext.detok
perl /mnt/matylda3/vydana/HOW2_EXP/MOSES_DECODER/mosesdecoder/scripts/tokenizer/detokenizer.perl -l ${tgt_lang} -q < hyp_utttext.detok1 > hyp_utttext.detok
#cat ref_utttext.detok1| sed -e "s/  //g" -e "s/(/ (/" -e "s/<space>/ /g" -e "s/>/> /g">ref_utttext.detok
#cat hyp_utttext.detok1| sed -e "s/  //g" -e "s/(/ (/" -e "s/<space>/ /g" -e "s/>/> /g">hyp_utttext.detok
perl /mnt/matylda3/vydana/HOW2_EXP/MOSES_DECODER/mosesdecoder/scripts/generic/multi-bleu-detok.perl ref_utttext.detok < hyp_utttext.detok > "$hyp_file".dtok_score
perl /mnt/matylda3/vydana/HOW2_EXP/MOSES_DECODER/mosesdecoder/scripts/generic/multi-bleu-detok.perl -lc ref_utttext.detok < hyp_utttext.detok > "$hyp_file".dtok_score_lc

#more "$hyp_file".dtok_score*
printf "%s\n" ""$hyp_file".dtok_score: $(<"$hyp_file".dtok_score)"
printf "%s\n" ""$hyp_file".dtok_score_lc: $(<"$hyp_file".dtok_score_lc)"
#more "$hyp_file".dtok_score_lc

}

