. /home/gs534/rds/hpc-work/work/espnet/tools/anaconda/etc/profile.d/conda.sh && conda deactivate && conda activate espnet
expdir=exp/finetune_librispeech_lr0.0005_KB200_drop0.1
testset=other
decodedir=decode_no_lm_b50_KB1000_${testset}_50best
mkdir -p $expdir/$decodedir
python decode.py \
    --test_json data/LibriSpeech/test_${testset}_full.json \
    --beamsize 50 \
    --expdir $expdir/$decodedir \
    --loadfrom $expdir/model.acc.best \
    --biasing \
    --biasinglist data/LibriSpeech/Blist/all_rare_words.txt \
    --dropentry 0.0 \
    --maxKBlen 1000 \
    --save_nbest \
    # --use_gpt2 \
    # --lm_weight 0.01 \
    # --ilm_weight 0.005 \
