#!/usr/bin/env python
# -------------------------------------------------------
# Joo-Kyung Kim (kimjook@cse.ohio-state.edu)	3/5/2017
# -------------------------------------------------------

import sys, argparse

parser = argparse.ArgumentParser()

wordTags = []
for line in sys.stdin:
	if line[0] == '#':
		continue
	tabs = line.strip().split('\t')
	if len(tabs) == 1 and len(tabs[0]) == 0:
		for p in wordTags:
			print(p[0] + '\t' + p[1])
		print()
		wordTags = []
	else:
		wordTags.append((tabs[1], tabs[3]))
