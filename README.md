# Sequence Tagging with Cross-Lingual Transfer Learning
The codes were used for the experiments in the paper:

Joo-Kyung Kim, Young-Bum Kim, Ruhi Sarikaya, and Eric Fosler-Lussier, "Cross-lingual Transfer Learning for POS Tagging without Cross-Lingual Resources," EMNLP 2017.

The codes were tested with PyTorch 0.1.12 on Python 3.6. (They are not compatible with Python 2.x but can be migrated easily.)


##	Scripts for the data preparation
Refer to files in `ud-treebanks-v1.4`. Only files for English exist in the repository. The files for other languages can be generated with the following scripts.

`chg_corpus_form.py`: Change the UD corpus files (https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-1827) to the usable formats in the codes.

`gen_shuffled_trainset.py`: Used to shuffle the trainset file

`gen_idxs.py`: Generate word, character, and tag index files


##	Main executable
`main_sepEmbSepTags.py`: The main executable file. It has an individual word embedding for each language but shares a label space.
Execution example:
```
./main_sepEmbSepTags.py --tag --tag_acc_flag --lang en_hu
```


##	Major options for main_sepEmbSepTags.py (Refer to the file for the detailed options)
--corpus_dir: the directory containing all the corpus related files (train, dev, test, word_idx, char_idx, and tag_idx)

--lang: Underscore separated languages (e.g., en_hu if using English and Hungarian)

--tag: Sequence tagging

--tag_acc_flag: If True, evaluate with accuracies. If False, with conll script (F1)

--no_shared_rnns: Do not use the shared RNN (Similar to Yang17's model)

--no_sep_rnns: Do not use the separate RNNs (Similar to Jaech16's model)

--rnn_merge_concat: Concat shared RNN outputs and separate RNN outputs. (default is summation)

--utt_enc_type: An utterance is represented as a single vector by 0: sum, 1: mean, 2: CNN w/ avg pooling, 3: CNN w/ max pooling

--dm_adv: 0: nothing, 1: Adversarial training on the shared RNN w/ single (K+1)-ary language classifier, 2: ~~ w/ K binary language classifiers, 3: ~~ w/ K+1 binary classifiers

--recon: bidirectional language modeling

--tag_first_train_mb_cnt, --tag_mid_train_mb_cnt, --tag_last_train_mb_cnt: Limit the number of minibaches, where the POS tag labels will be utilized in the training for the first, those in the middle, or the last language, respectively.

--pivot_weights: Set the weight of the target language to be 1 and set others to keep the relative proportions to be the same

--dm_tag_weights: Weights to the tagging loss

--dm_disc_crit_weights: Set language classifier criterion weights inversely proportional to # minibatches per each language. If False, all the weights are set to 1