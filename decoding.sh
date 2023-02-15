. /home/gs534/rds/hpc-work/work/espnet/tools/anaconda/etc/profile.d/conda.sh && conda deactivate && conda activate espnet
expdir=exp/finetune_librispeech_lr0.0005_KB100_drop0.3
decodedir=decode_no_lm_b10_KB1000
mkdir -p $expdir/$decodedir
python decode.py \
    --test_json data/LibriSpeech/test_clean_full.json \
    --beamsize 10 \
    --expdir $expdir/$decodedir \
    --loadfrom $expdir/model.acc.best \
    --biasing \
    --biasinglist data/LibriSpeech/Blist/all_rare_words.txt \
    --dropentry 0.0 \
    --maxKBlen 1000 \
