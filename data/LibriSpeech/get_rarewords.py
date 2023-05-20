import sys, os
import json


setname = "test_other"

with open("{}.json".format(setname)) as fin:
    data = json.load(fin)

with open("Blist/all_rare_words.txt") as fin:
    rarewords = [word.strip() for word in fin]

for uttname, utt in data.items():
    uttKB = []
    for word in utt["words"].split():
        if word in rarewords and word not in uttKB:
            uttKB.append(word)
    data[uttname]["blist"] = uttKB

with open("{}_full.json".format(setname), "w") as fout:
    json.dump(data, fout, indent=4)
