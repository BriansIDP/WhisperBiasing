import sys, os

error_words_freqs = {}
infile = sys.argv[1]
# setname = sys.argv[2]
insert_error = 0
insert_rare = 0
freqlist_test = {}

freqlist = {}
with open("word_freq.txt") as fin:
    for line in fin:
        word, freq = line.split()
        freqlist[word.upper()] = int(freq)

# with open("../data/SLURP/Blist/rarewords_f30.txt") as fin:
with open("../data/LibriSpeech/Blist/all_rare_words.txt") as fin:
# with open("/home/dawna/gs534/espnet/egs/librispeech/asr1/data/KBs/KBfull_list_chapter_{}.txt".format(setname)) as fin:
    rareset = set()
    for line in fin:
        rareset.add(line.strip().upper())

project_set = set()
with open(infile) as fin:
    lines = fin.readlines()
for i, line in enumerate(lines):
    if line.startswith('id:'):
        project = line.strip(')\n').split('-')[-3:]
        project = '-'.join(project)
    if "REF:" in line:
        nextline = lines[i+1].split()
        for j, word in enumerate(line.split()):
            if '*' in word:
                insert_error += 1
                if nextline[j].upper() in rareset:
                    insert_rare += 1
        line = line.replace('*', '')
        line.replace('%BCACK', '')
        for word in line.split()[1:]:
            if not word.startswith('('):
                if word.upper() not in freqlist_test:
                    freqlist_test[word.upper()] = 1
                else:
                    freqlist_test[word.upper()] += 1

                if word != word.lower() and word.upper() in error_words_freqs:
                    error_words_freqs[word.upper()] += 1
                elif word != word.lower() and word.upper() not in error_words_freqs:
                    error_words_freqs[word.upper()] = 1
                elif word == word.lower() and word.upper() not in error_words_freqs:
                    error_words_freqs[word.upper()] = 0

                # if word != word.lower() and word in freqlist and freqlist[word.upper()] >= 3 and freqlist[word.upper()] <= 10:
                #     if project not in project_set:
                #         project_set.add(project)    
                        
print(len(error_words_freqs.keys()))
print(insert_rare)

# with open('project_set.txt', 'w') as fout:
#     for project in project_set:
#         fout.write('{}\n'.format(project))
# with open('error_words.txt', 'w') as fout:
#     for word, error in error_words_freqs.items():
#         fout.write('{} {}\n'.format(word, error))

commonwords = []
rarewords = []
oovwords = []
common_freq = 0
rare_freq = 0
oov_freq = 0
common_error = 0
rare_error = 0
oov_error = 0
partial_error = 0
partial_freq = 0
very_common_error = 0
very_common_words = 0
words_error_freq = {}
words_total_freq = {}
for word, error in error_words_freqs.items():
    if word in rareset:
        rarewords.append(word)
        rare_freq += freqlist_test[word]
        rare_error += error
    elif word not in freqlist:
        oovwords.append(word)
        oov_freq += freqlist_test[word] if word in freqlist_test else 1
        oov_error += error
    else:
        if freqlist[word] <= 10 and freqlist[word] >= 3:
            if freqlist[word] not in words_error_freq:
                words_error_freq[freqlist[word]] = error
                words_total_freq[freqlist[word]] = freqlist_test[word]
            else:
                words_error_freq[freqlist[word]] += error
                words_total_freq[freqlist[word]] += freqlist_test[word]
        if freqlist[word] <= 10 and freqlist[word] >= 3:
            very_common_error += error
            very_common_words += freqlist_test[word]
        commonwords.append(word)
        common_freq += freqlist_test[word]
        common_error += error

# with open('word_error_list.txt', 'w') as fout:
#     for word, error in error_words_freqs.items():
#         if word in freqlist_test:
#             error_rate = error/freqlist_test[word]
#             fout.write('{}\t{}\t{}\t{:.2f}\n'.format(word, error, freqlist_test[word], error/freqlist_test[word]))
total_words = common_freq + rare_freq + oov_freq
total_errors = common_error+rare_error+oov_error + insert_error
WER = total_errors / total_words
print('='*89)
print('Common words error freq: {} / {} = {}'.format(common_error, common_freq, common_error/common_freq))
print('Rare words error freq: {} / {} = {}'.format(rare_error+insert_rare, rare_freq, (rare_error + insert_rare)/rare_freq))
print('OOV words error freq: {} / {} = {}'.format(oov_error, oov_freq, oov_error/max(oov_freq, 1)))
print('WER estimate: {} / {} = {}'.format(total_errors, total_words, WER))
# print('Partial word count: {} / {}'.format(partial_error, partial_freq))
print('Insert error: {} / {} = {}'.format(insert_error - insert_rare, total_words, (insert_error - insert_rare)/total_words))
print('Insertion + OOV error {}'.format((insert_error + oov_error - insert_rare) / total_words))
# print('Very common words error freq: {} / {} = {}'.format(very_common_error, very_common_words, very_common_error/very_common_words))
print('='*89)

# with open('freq_error_data.txt', 'w') as fout:
#     fout.write('frequency error total\n')
#     for freq in range(3, 101):
#         fout.write('{} {} {}\n'.format(freq, words_error_freq[freq], words_total_freq[freq]))
