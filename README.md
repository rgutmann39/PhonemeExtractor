# PhonemeExtractor
Extracts phoneme sequences from speech audio files


## Text2Phoneme experiments
All the expirements conducted for text2phoneme can be found in the `text2phoneme_experiments`
directory. The expreiments are in the form of jupyter notebooks that can be run with no additional
setup necessary other than pointing certain cells to your correct directories. Special thanks to 
[Ben Trevett](https://github.com/bentrevett/pytorch-seq2seq) and his awesome seq2seq modeling
tutorials for helping us get our experiments working.

`text2phoneme_experiments/DeepPhonemizer` contains the modified grapheme2phoneme experiment for
timit asr dataset which serves as our baseline.

## Phoneme Sequence Matching algorithm
In `matching_algs.py` you'll find the `phoneme_match` function which idenitifies mispredicted words
given the gt and reference phonemes + their corresponding word mappings. The test cases to ensure
the correctness of this algorithm are also present in this file. It can be run off the shelf as
follows:
```bash
$ python matching_algs.py
```

## Audio to Phoneme Model
Use the 'audio_to_phoneme.py' file to train a feature extractor, tokenizer, and model from scratch for converting wav audio files into phoneme sequences.
Make sure to have python version 3.8.7 installed and then run the following two shell commands:
```bash
$ pip install datasets transformers jiwer soundfile torch librosa dataclasses typing
$ python audio_to_phoneme.py
```
