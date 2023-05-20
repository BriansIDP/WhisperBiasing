import sys, os
import json

import torch
import whisper


setname = "train-clean-100"
tokenizer = whisper.tokenizer.get_tokenizer(True, language="en")

features = {}
for speaker in os.listdir(setname):
    spkpath = os.path.join(setname, speaker)
    for project in os.listdir(spkpath):
        fullpath = os.path.join(spkpath, project)
        with open(os.path.join(fullpath, "{}-{}.trans.txt".format(speaker, project))) as fin:
            for line in fin:
                uttname = line.split()[0]
                print(uttname)
                utt = " " + ' '.join(line.split()[1:])
                utttokens = tokenizer.encode(utt.lower())
                audiopath = os.path.join("/scratch/whisper/data/LibriSpeech", fullpath, "{}.flac".format(uttname))
                audio = whisper.load_audio(audiopath)
                audio = whisper.pad_or_trim(audio)
                mel = whisper.log_mel_spectrogram(audio)
                dumppath = os.path.join("/scratch/whisper/data/LibriSpeech", fullpath, "{}_fbank.pt".format(uttname))
                torch.save(mel, dumppath)
                datapiece = {"fbank": dumppath, "words": utt}
                features[uttname] = datapiece

with open(setname.replace("-", "_")+".json", "w") as fout:
    json.dump(features, fout, indent=4)
