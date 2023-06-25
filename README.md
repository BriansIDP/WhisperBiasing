# Tree-Constrained Pointer Generator (TCPGen) for Whisper Biasing

## Whisper Biasing

[[Paper]](https://arxiv.org/pdf/2306.01942.pdf)

End-to-end automatic speech recognition (ASR) and large language models, such as Whisper and GPT-2, have recently been
scaled to use vast amounts of training data. Despite a large amount of training data, infrequent content words that occur in
a particular task may still exhibit poor ASR performance, with contextual biasing a possible remedy. This paper investigates
the effectiveness of neural contextual biasing for Whisper combined with GPT-2. Specifically, this paper proposes integrating
an adapted tree-constrained pointer generator (TCPGen) component for Whisper and a dedicated training scheme to dynamically adjust the final output without modifying any Whisper
model parameters. Experiments across three datasets show a
considerable reduction in errors on biasing words with a biasing list of 1000 words. Contextual biasing was more effective
when applied to domain-specific data and can boost the performance of Whisper and GPT-2 without losing their generality.

## Dependencies
All required packages for Whisper

## Data and biasing list preparation
We use LibriSpeech as an example, but this can be applied to SLURP and DSTC as well.
1. Dump features
```
cd data/LibriSpeech
python dump_feature.py
```
Note that you need to change `setname='train-clean-100'` to the set you want.

2. Biasing lists
Biasing lists are already prepared:

`rareword_error.txt`: error-based biasing list for training

`all_rare_words.txt`: full biasing list for inference

Use `get_rarewords.py` to get JSON data files containing per-utterance biasing words, e.g. `train_clean_100_error.json` which is used for training.

## Training
run training script `train_large.sh` for training. 

## Decoding
run decoding script `decoding.sh` for decoding.

## Scoring
score with `score.sh` after decoding.
Use `error_analysis/get_error_word_count.py` to calculate R-WER, by passing `<path_to_results.txt>` as the argument to it.

## Expected results (test-clean)
| System      | WER         | R-WER       |
| ----------- | ----------- | ----------- |
| Whisper large unnormalised      | 4.0%       |  10.4%         |
| Whisper large + TCPGen unnormalised   | 3.4%        |     8.3%   |
| Whisper large normalised      | 2.5%       |  8.1%         |
| Whisper large + TCPGen normalised   | 2.3%        |     7.0%   |
