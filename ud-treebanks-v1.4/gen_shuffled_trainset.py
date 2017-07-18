#!/usr/bin/env python
# -------------------------------------------------------
# Joo-Kyung Kim (kimjook@cse.ohio-state.edu)	4/8/2017
# -------------------------------------------------------

import sys, argparse, random

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=1000, help='Seed for random numbers')
args = parser.parse_args()

random.seed(args.seed)

sents = []
wordTags = []
for line in sys.stdin:
	tabs = line.strip().split('\t')
	if len(tabs) == 1 and len(tabs[0]) == 0:
		sents.append(wordTags)
		wordTags = []
	else:
		wordTags.append((tabs[0], tabs[1]))


random.shuffle(sents)	# Shuffle the trainset sentences

for sent in sents:
	for token in sent:
		print(token[0] + '\t' + token[1])
	print()
