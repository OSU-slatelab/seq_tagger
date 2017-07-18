#!/usr/bin/env python
# -------------------------------------------------------
# Joo-Kyung Kim (kimjook@cse.ohio-state.edu)	3/22/2017
# -------------------------------------------------------

import sys, argparse, re

parser = argparse.ArgumentParser()
parser.add_argument('--train_file', type=str)
parser.add_argument('--dev_file', type=str, default='', help='If provided, even if # occurs of a word in the trainset is below in_cnt, if it was observed in the devset, include it.')
parser.add_argument('--char_idx_file', type=str, default='')
parser.add_argument('--word_idx_file', type=str, default='')
parser.add_argument('--tag_idx_file', type=str, default='')
parser.add_argument('--min_cnt', type=int, default=2, help='words occuring less than this is regarded as <unk>')
args = parser.parse_args()

charIdxs = set()
wordIdxs = {}
tagIdxs = set()

for line in open(args.train_file):
	tabs = line.strip().split('\t')
	if len(tabs) == 1 and len(tabs[0]) == 0:
		continue
	else:
		tabs[0] = re.sub('[0-9]', '0', tabs[0])	# change all the digits to 0
		for c in tabs[0]:
			charIdxs.add(c)
		if tabs[0] not in wordIdxs:
			wordIdxs[tabs[0]] = 1
		else:
			wordIdxs[tabs[0]] += 1
		tagIdxs.add(tabs[1])

devWordSet = set()

if len(args.dev_file) > 0:
	for line in open(args.dev_file):
		tabs = line.strip().split('\t')
		if len(tabs) == 1 and len(tabs[0]) == 0:
			continue
		else:
			tabs[0] = re.sub('[0-9]', '0', tabs[0])	# change all the digits to 0
			devWordSet.add(tabs[0])

if len(args.char_idx_file) > 0:
	charIdxFile = open(args.char_idx_file, 'w')
	charIdxFile.write('<unk>\n<w>\n</w>\n')
	for c in sorted(charIdxs):
		charIdxFile.write(c + '\n')
	charIdxFile.close()

if len(args.word_idx_file) > 0:
	wordIdxFile = open(args.word_idx_file, 'w')
	wordIdxFile.write('<unk>\n')
	wordIdxList = []
	for k,v in iter(wordIdxs.items()):
		if v >= args.min_cnt or k in devWordSet:
			wordIdxList.append(k)

	for word in sorted(wordIdxList):
		wordIdxFile.write(word + '\n')
	wordIdxFile.close()

if len(args.tag_idx_file) > 0:
	tagIdxFile = open(args.tag_idx_file, 'w')
	for tag in sorted(tagIdxs):
		tagIdxFile.write(tag + '\n')
	tagIdxFile.close()
