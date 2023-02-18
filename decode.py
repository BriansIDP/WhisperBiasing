import sys, os
import re
import time
import argparse

import torch
import whisper
import editdistance
from dataloader import get_dataloader, BiasingProcessor
from whisper.model import WhisperBiasing
from transformers import GPT2Tokenizer, GPT2Model
from whisper.normalizers.english import EnglishTextNormalizer

parser = argparse.ArgumentParser(description = 'Running Whisper experiments')

# set arguments for training and decoding. 
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--test_json', type=str, default="data/LibriSpeech/test_clean.json")
parser.add_argument('--beamsize', type=int, default=3)
parser.add_argument('--eval_batch_size', type=int, default=1)
parser.add_argument('--expdir', type=str, default="exp/origmodel")
parser.add_argument('--loadfrom', type=str, default="")
parser.add_argument('--biasing', action="store_true")
parser.add_argument('--use_gpt2', action="store_true")
parser.add_argument('--deepbiasing', action="store_true")
parser.add_argument('--attndim', type=int, default=256)
parser.add_argument('--biasinglist', type=str, default="data/LibriSpeech/Blist/rareword_f15.txt")
parser.add_argument('--maxKBlen', type=int, default=1)
parser.add_argument('--dropentry', type=float, default=0.0)
args = parser.parse_args()


if args.loadfrom != '':
    biasing_model = torch.load(args.loadfrom)
    biasing_model.eval()
    model = biasing_model.whisper
    useGPT = biasing_model.useGPT
    shallowfusion = args.use_gpt2
    GPTmodel = None
    if useGPT or args.use_gpt2:
        GPTmodel = GPT2Model.from_pretrained('gpt2').to(model.device)
        GPThiddim = GPTmodel.config.n_embd
else:
    model = whisper.load_model("base.en").eval()
    biasing_model = None
tokenizer = whisper.tokenizer.get_tokenizer(model.is_multilingual, language="en")

####################
# Data Loader
####################
testloader = get_dataloader(
    args.test_json,
    args.eval_batch_size,
    loadtarget=False,
    tokenizer=tokenizer,
    biasing=args.biasing,
    shuffle=False,
)
biasproc = BiasingProcessor(tokenizer, args.biasinglist, ndistractors=args.maxKBlen, drop=args.dropentry)
eng_norm = EnglishTextNormalizer()

totalwords = 0
totalwer = 0
total_hyp = []
total_ref = []

print("Start of decoding")
start = time.time()
for idx, data in enumerate(testloader):
    uttnames, fbank, tgt, blist = data
    fbank = fbank.to(model.device)
    origtree = biasproc.get_lextree(blist)

    if biasing_model is not None and getattr(biasing_model, "GNN", None) is not None:
        biasing_model.GNN(origtree, model.decoder.token_embedding)

    options = whisper.DecodingOptions(
        language="en",
        without_timestamps=True,
        beam_size=args.beamsize,
        biasing=args.biasing,
        biasingmodule=biasing_model,
        origtree=origtree,
        fp16=False,
        shallowfusion=shallowfusion,
        useGPT=useGPT,
        GPT2=GPTmodel,
    )
    result = whisper.decode(model, fbank, options)
    for i, utt in enumerate(tgt):
        uttname = uttnames[i]
        text = result[i].text.lower()
        text = eng_norm(text).split()
        refwords = eng_norm(utt).split()
        we = editdistance.eval(text, refwords)
        totalwords += len(refwords)
        totalwer += we
        fulltext = "{} ({})\n".format(' '.join(text), uttname)
        fullref = "{} ({})\n".format(utt.lower(), uttname)
        total_hyp.append(fulltext)
        total_ref.append(fullref)
    if idx % 10 == 0 and idx > 0:
        print("{} out of {} finished | time elapsed {}".format(idx, len(testloader), time.time()-start))
        print("WER: {}/{}={}".format(totalwer, totalwords, totalwer/totalwords))

print("WER: {}/{}={}".format(totalwer, totalwords, totalwer/totalwords))

with open(os.path.join(args.expdir, "hyp.wrd.trn"), "w") as fout:
    for line in total_hyp:
        fout.write(line + '\n')
with open(os.path.join(args.expdir, "ref.wrd.trn"), "w") as fout:
    for line in total_ref:
        fout.write(line + '\n')
