import sys, os
import re
import time
import argparse
import json

import torch
import whisper
import editdistance
from dataloader import get_dataloader, BiasingProcessor
from whisper.model import WhisperBiasing
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
from whisper.normalizers.english import EnglishTextNormalizer

parser = argparse.ArgumentParser(description = 'Running Whisper experiments')

# set arguments for training and decoding. 
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--test_json', type=str, default="data/LibriSpeech/test_clean.json")
parser.add_argument('--device', type=str, default="cuda:0")
parser.add_argument('--beamsize', type=int, default=3)
parser.add_argument('--eval_batch_size', type=int, default=1)
parser.add_argument('--expdir', type=str, default="exp/origmodel")
parser.add_argument('--loadfrom', type=str, default="")
parser.add_argument('--biasing', action="store_true")
parser.add_argument('--use_gpt2', action="store_true")
parser.add_argument('--save_nbest', action="store_true")
parser.add_argument('--lm_weight', type=float, default=0)
parser.add_argument('--ilm_weight', type=float, default=0)
parser.add_argument('--deepbiasing', action="store_true")
parser.add_argument('--attndim', type=int, default=256)
parser.add_argument('--biasinglist', type=str, default="data/LibriSpeech/Blist/rareword_f15.txt")
parser.add_argument('--maxKBlen', type=int, default=1)
parser.add_argument('--dropentry', type=float, default=0.0)
parser.add_argument('--modeltype', type=str, default="base.en")
parser.add_argument('--normalise', action="store_true")
parser.add_argument('--logfile', type=str, default="")
args = parser.parse_args()


def logging(s, logfile, logging_=True, log_=True):
    print(s)
    if log_:
        with open(logfile, 'a+') as f_log:
            f_log.write(s + '\n')

shallowfusion = args.use_gpt2
useGPT = None
GPTtokenizer = None
normaliser = EnglishTextNormalizer()
logfile = args.logfile if args.logfile != "" else os.path.join(args.expdir, "log.txt")
if args.use_gpt2:
    GPTmodel = GPT2LMHeadModel.from_pretrained('gpt2', output_hidden_states=True).to(args.device)
    GPThiddim = GPTmodel.config.n_embd
else:
    GPTmodel = None

if args.loadfrom != '':
    biasing_model = torch.load(args.loadfrom)
    biasing_model.eval()
    model = biasing_model.whisper
    useGPT = getattr(biasing_model, "useGPT", False)
    if useGPT or args.use_gpt2:
        GPTtokenizer = GPT2Tokenizer.from_pretrained('gpt2')
else:
    model = whisper.load_model(args.modeltype).eval()
    biasing_model = None
    useGPT = False

ilme_model = None
if args.use_gpt2 and args.ilm_weight > 0:
    ilme_model = whisper.load_model("base.en").eval()
shallowfusion = args.use_gpt2
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

totalwords = 0
totalwer = 0
total_hyp = []
total_ref = []
nbest_dict = {}

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
        lm_weight=args.lm_weight,
        GPT2tokenizer=GPTtokenizer,
        ilm_weight=args.ilm_weight,
        ilme_model=ilme_model,
    )
    result = whisper.decode(model, fbank, options)
    for i, utt in enumerate(tgt):
        uttname = uttnames[i]
        if args.normalise:
            text = normaliser(result[i].text).split()
            refwords = normaliser(utt.lower()).split()
        else:
            text = result[i].text.lower()
            text = re.sub("[^a-zA-Z\' ]+", "", text).split()
            refwords = utt.lower().split()
        we = editdistance.eval(text, refwords)
        totalwords += len(refwords)
        totalwer += we
        fulltext = "{} ({})\n".format(' '.join(text), uttname)
        fullref = "{} ({})\n".format(normaliser(utt.lower()) if args.normalise else utt.lower(), uttname)
        total_hyp.append(fulltext)
        total_ref.append(fullref)
        if args.save_nbest:
            text_nbest = [text_nbest_i.lower() for text_nbest_i in result[i].text_nbest]
            text_nbest = [re.sub("[^a-zA-Z\' ]+", "", text_nbest_i) for text_nbest_i in text_nbest]
            sum_logprob_nbest = result[i].sum_logprob_nbest
            token_nbest = result[i].token_nbest
            nbest_dict[uttname] = [
                {"text": t, "token": token, "whisper_slp": slp}
                for t, slp, token in zip(text_nbest, sum_logprob_nbest, token_nbest)
            ]

    if idx % 10 == 0 and idx > 0:
        print("{} out of {} finished | time elapsed {}".format(idx, len(testloader), time.time()-start))
        print("WER: {}/{}={}".format(totalwer, totalwords, totalwer/totalwords))
        logging("{} out of {} finished | time elapsed {} | WER: {}".format(
            idx, len(testloader), time.time()-start, totalwer/totalwords), logfile)

print("WER: {}/{}={}".format(totalwer, totalwords, totalwer/totalwords))

with open(os.path.join(args.expdir, "hyp.wrd.trn"), "w") as fout:
    for line in total_hyp:
        fout.write(line + '\n')
with open(os.path.join(args.expdir, "ref.wrd.trn"), "w") as fout:
    for line in total_ref:
        fout.write(line + '\n')

if args.save_nbest:
    with open(os.path.join(args.expdir, "nbest.json"), "w") as fout:
        json.dump(nbest_dict, fout)
