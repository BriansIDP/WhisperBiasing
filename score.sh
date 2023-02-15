expdir=$1
/home/gs534/rds/hpc-work/work/espnet/tools/sctk/bin/sclite -r $expdir/ref.wrd.trn trn -h $expdir/hyp.wrd.trn trn -i rm -o all stdout > $expdir/results.txt
