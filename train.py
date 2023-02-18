import sys, os
import re
import time

import whisper
from whisper.model import WhisperBiasing
import editdistance
from dataloader import get_dataloader, BiasingProcessor
import argparse
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.optim import SGD, Adam
from transformers import WhisperTokenizer
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel

parser = argparse.ArgumentParser(description = 'Running Whisper experiments')

# set arguments for training and decoding. 
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--modeltype', type=str, default="base.en")
parser.add_argument('--train_json', type=str, default="data/LibriSpeech/train_clean_100.json")
parser.add_argument('--dev_json', type=str, default="data/LibriSpeech/dev.json")
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--nepochs', type=int, default=10)
parser.add_argument('--expdir', type=str, default="exp/origmodel")
parser.add_argument('--lr', type=float, default=0.00001)
parser.add_argument('--decay_pct', type=float, default=1)
parser.add_argument('--warmup_pct', type=float, default=0.0)
parser.add_argument('--log_interval', type=int, default=200)
parser.add_argument('--logfile', type=str, default="log")
parser.add_argument('--accumgrad', type=int, default=1)
parser.add_argument('--biasing', action="store_true")
parser.add_argument('--biasinglist', type=str, default="data/LibriSpeech/Blist/rareword_f15.txt")
parser.add_argument('--maxKBlen', type=int, default=1)
parser.add_argument('--dropentry', type=float, default=0.0)
parser.add_argument('--attndim', type=int, default=256)
parser.add_argument('--loadfrom', type=str, default="")
parser.add_argument('--GNNtype', type=str, default="none")
parser.add_argument('--GNNdim', type=int, default=0)
parser.add_argument('--useGPT', action="store_true")
args = parser.parse_args()

def logging(s, logfile, logging_=True, log_=True):
    print(s)
    if log_:
        with open(logfile, 'a+') as f_log:
            f_log.write(s + '\n')


##################
# Model
##################
torch.manual_seed(args.seed)
if args.loadfrom != "":
    whisperbiasing = torch.load(args.loadfrom)
    model = whisperbiasing.whisper
else:
    model = whisper.load_model(args.modeltype)
model.train()
if args.useGPT:
    # GPTmodel = GPT2Model.from_pretrained('gpt2').to(model.device)
    GPTmodel = GPT2LMHeadModel.from_pretrained('gpt2', output_hidden_states=True).to(model.device)
    GPThiddim = GPTmodel.config.n_embd
    GPTtokenizer = GPT2Tokenizer.from_pretrained('gpt2')
else:
    GPThiddim = 0

options = whisper.DecodingOptions(language="en", fp16=False, without_timestamps=True)
tokenizer = whisper.tokenizer.get_tokenizer(model.is_multilingual, language="en")
decodetask = whisper.decoding.DecodingTask(model, options)
logit_filters = decodetask.logit_filters
sot_sequence = decodetask.sot_sequence
sotlen = len(sot_sequence)
if args.loadfrom == "":
    whisperbiasing = WhisperBiasing(
        model,
        tokenizer,
        model.dims.n_text_state,
        model.dims.n_text_state,
        args.attndim,
        model.dims.n_vocab,
        Bdrop=0.1,
        biasing=args.biasing,
        GNNtype=args.GNNtype,
        GNNdim=args.GNNdim,
        useGPT=args.useGPT,
        GPThiddim=GPThiddim,
    ).to(model.device)
whisperbiasing.train()

##################
# Data Loader
##################
trainloader = get_dataloader(args.train_json, args.batch_size, loadtarget=True, tokenizer=tokenizer, biasing=args.biasing)
devloader = get_dataloader(args.dev_json, args.batch_size, loadtarget=True, tokenizer=tokenizer, biasing=args.biasing)
biasproc = BiasingProcessor(tokenizer, args.biasinglist, ndistractors=args.maxKBlen, drop=args.dropentry)

##################
# Training
##################
criterion = torch.nn.NLLLoss()
optimiser = Adam(whisperbiasing.parameters(), lr=args.lr)

##################
# Start Training
##################
logging("Start of training", args.logfile)
bestacc = 0
for epoch in range(args.nepochs):
    start = time.time()
    totalloss = 0
    for idx, data in enumerate(trainloader):
        uttnames, fbank, tgt, blist = data
        lextree = biasproc.get_lextree(blist)
        fbank = fbank.to(model.device)
        origtarget = [torch.tensor(list(sot_sequence) + y, dtype=torch.long) for y in tgt]
        GPT_last_hidden = None
        GPT_distribution = None
        target = pad_sequence(origtarget, batch_first=True, padding_value=-100).to(model.device)
        targetmask = target != -100
        if args.useGPT:
            with torch.no_grad():
                # Replace Whisper bos token with GPT2 bos token
                GPTtarget_ids = (target*targetmask)[:, sotlen-1:-1]
                GPTtarget_ids[:, 0] = GPTtokenizer.bos_token_id
                GPTtarget = {"input_ids": GPTtarget_ids, "attention_mask": targetmask[:, sotlen-1:-1]}

                # Get GPT states
                GPToutputs = GPTmodel(**GPTtarget)
                GPT_last_hidden = GPToutputs.hidden_states[-1]
                GPT_distribution = torch.softmax(GPToutputs.logits, dim=-1)

                # Need to pad GPT2 distribution to be the same vocab size as Whisper distribution using zero padding
                zeropadding_dist = GPT_distribution.new_zeros(GPT_distribution.size(0), GPT_distribution.size(1),
                    whisperbiasing.nvocab-GPT_distribution.size(2))
                GPT_distribution = torch.cat([GPT_distribution, zeropadding_dist], dim=-1)

                # Need to pad the sequence with zeros
                zeropadding = torch.zeros(GPT_last_hidden.size(0), 1, GPT_last_hidden.size(-1)).to(model.device)
                GPT_last_hidden = torch.cat([zeropadding for _ in range(sotlen-1)] + [GPT_last_hidden, zeropadding], dim=1)

        optimiser.zero_grad()

        # Forward the biasing model
        loss, p_final = whisperbiasing(fbank, target, targetmask, lextree, sotlen,
            GPThidden=(GPT_last_hidden, GPT_distribution))
        loss = loss / args.accumgrad

        loss.backward()
        totalloss += loss.item()
        if idx != 0 and idx % args.accumgrad == 0:
            # LR scheduler
            currentstep = epoch * len(trainloader) + idx + 1
            totalstep = args.nepochs * len(trainloader)
            if currentstep > int(args.decay_pct * totalstep):
                factor = (totalstep - currentstep) / (totalstep - int(args.decay_pct * totalstep))
                optimiser.param_groups[0]['lr'] = args.lr * max(0, factor)
            elif currentstep < int(args.warmup_pct * totalstep):
                factor = currentstep / int(args.warmup_pct * totalstep)
                optimiser.param_groups[0]['lr'] = args.lr * factor
            optimiser.step()

        if idx != 0 and idx % args.log_interval == 0:
            logging("{} / {} steps finished in {} | Loss: {} | lr: {}".format(
                idx, len(trainloader), time.time()-start, totalloss/args.log_interval, optimiser.param_groups[0]['lr']),
                 args.logfile)
            totalloss = 0

    # Validation
    totalvalset = 0
    totalvalacc = 0
    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(devloader):
            uttnames, fbank, tgt, blist = data
            lextree = biasproc.get_lextree(blist)
            fbank = fbank.to(model.device)
            target = [torch.tensor(list(sot_sequence) + y, dtype=torch.long) for y in tgt]
            # target = [torch.tensor(y, dtype=torch.long) for y in tgt]
            target = pad_sequence(target, batch_first=True, padding_value=-100).to(model.device)
            targetmask = target != -100
            if args.useGPT:
                # Replace Whisper bos token with GPT2 bos token
                GPTtarget_ids = (target*targetmask)[:, sotlen-1:-1]
                GPTtarget_ids[:, 0] = GPTtokenizer.bos_token_id
                GPTtarget = {"input_ids": GPTtarget_ids, "attention_mask": targetmask[:, sotlen-1:-1]}

                # Get GPT states
                GPToutputs = GPTmodel(**GPTtarget)
                GPT_last_hidden = GPToutputs.hidden_states[-1]
                GPT_distribution = torch.softmax(GPToutputs.logits, dim=-1)

                # Need to pad GPT2 distribution to be the same vocab size as Whisper distribution using zero padding
                zeropadding_dist = GPT_distribution.new_zeros(GPT_distribution.size(0), GPT_distribution.size(1),
                    whisperbiasing.nvocab-GPT_distribution.size(2))
                GPT_distribution = torch.cat([GPT_distribution, zeropadding_dist], dim=-1)

                # Need to pad the sequence with zeros
                zeropadding = torch.zeros(GPT_last_hidden.size(0), 1, GPT_last_hidden.size(-1)).to(model.device)
                GPT_last_hidden = torch.cat([zeropadding for _ in range(sotlen-1)] + [GPT_last_hidden, zeropadding], dim=1)

            # Forward biasing model
            loss, output = whisperbiasing(fbank, target, targetmask, lextree, sotlen,
                GPThidden=(GPT_last_hidden, GPT_distribution))

            target = target[:, sotlen:]
            output = output.view(target.size(0), target.size(1), -1).max(dim=-1)[1]
            totalvalacc += ((output == target) * targetmask[:, sotlen:]).sum()
            totalvalset += targetmask[:, sotlen:].sum()

            # result = whisper.decode(model, fbank, options)
            if idx % 50 == 0 and idx > 0:
                logging("{} out of {} finished | time elapsed {} | ACC: {}".format(
                    idx, len(devloader), time.time()-start, totalvalacc/totalvalset), args.logfile)
        logging("Total ACC: {}".format(totalvalacc/totalvalset), args.logfile)

        totalacc = totalvalacc / totalvalset
    if totalacc > bestacc:
        torch.save(whisperbiasing, os.path.join(args.expdir, "model.acc.best"))
        bestacc = totalacc
        logging("Saving best model at epoch {}".format(epoch+1), args.logfile)
    torch.save(whisperbiasing, os.path.join(args.expdir, "snapshot.ep.{}".format(epoch+1)))
