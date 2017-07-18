#!/usr/bin/env python
# -------------------------------------------------------
# Joo-Kyung Kim (kimjook@cse.ohio-state.edu)	07/15/2017
# -------------------------------------------------------

import sys, argparse, tempfile, subprocess, os, math, time, random, re, shutil
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import util
from model_sepEmbSepTags import WordEmbModel
from model_sepEmbSepTags import SharedModel
from model_sepEmbSepTags import SepModel

parser = argparse.ArgumentParser()
parser.add_argument('--corpus_dir', type=str, default='./ud-treebanks-v1.4/', help='Corpus directory')
parser.add_argument('--langs', type=str, default='en', help='Underline-separated language abbreviations')
parser.add_argument('--init_word_vec', type=str, default='', help='Initial word vector file')
parser.add_argument('--emb_dim', type=int, default=128, help='Word embedding dimensionality')
parser.add_argument('--hid_dim', type=int, default=100, help='Hidden unit dimensionality')
parser.add_argument('--char_emb_dim', type=int, default=100, help='Word embedding dimensionality')
parser.add_argument('--char_hid_dim', type=int, default=100, help='Hidden unit dimensionality')
parser.add_argument('--conv_out_dim', type=int, default=64, help='CNN filter output dimensionality')
parser.add_argument('--conv_filters', type=str, default='3_4_5', help='Underline-separated convolution filter sizes')
parser.add_argument('--unk_word_zero', default=False, action='store_true', help='If True, let <unk> vector to be always a zero vector')
parser.add_argument('--word_dropout', type=float, default=0.5, help='Dropout rate after word embedding')
parser.add_argument('--rnn_dropout', type=float, default=0.5, help='Dropout rate after RNNs')
parser.add_argument('--no_shared_rnn', default=False, action='store_true', help='If True, shared RNN is not used')
parser.add_argument('--no_sep_rnns', default=False, action='store_true', help='If True, separate RNNs are not used (similar to Jaech16 model)')
parser.add_argument('--no_sep_for_pred', default=False, action='store_true', help='If True, separate RNNs are not merged for final prediction')
parser.add_argument('--rnn_prj', default=False, action='store_true', help='Add a linear projection after RNNs')
parser.add_argument('--rnn_merge_cat', default=False, action='store_true', help='If True, concat the shared RNN output and the separate RNN output. Otherwise, sum them')
parser.add_argument('--rnn_layer_cnt', default=1, type=int, help='RNN layer count')
parser.add_argument('--tag', default=False, action='store_true', help='Do sequence tagging')
parser.add_argument('--intent', default=False, action='store_true', help='Do intent classification. Currently not implemented')
parser.add_argument('--dm_tag_weights', default=False, action='store_true', help='Weights to the tagging loss')
parser.add_argument('--pivot_weights', default=False, action='store_true', help='Set the weight of the target domain to be 1 and set others to keep the relative proportions to be the same')
parser.add_argument('--attn', default=0, type=int, help='Use attention as [Liu and Lane, Interspeech 2016]. 0: no attention, 1: single attention after merging shared and separate representations, 2: Separate attentions (one for shared representations and the other for separate representations)')
parser.add_argument('--attn_winsize', default=0, type=int, help='Attention window size for each direction')
parser.add_argument('--utt_enc_type', default=0, type=int, help='0: sum, 1: mean, 2: CNN w/ avg-pooling, 3: CNN w/ max-pooling')
parser.add_argument('--utt_enc_noise', default=False, action='store_true', help='Add white noises to the utterance encoding (mean 0 and stdev 0.3)')
parser.add_argument('--utt_enc_bn', default=False, action='store_true', help='Batch normalization to the utterance encoding')
parser.add_argument('--dm_adv', default=0, type=int, help='0: no adv, 1: use single (K+1)-ary discriminator, 2: use K binary discriminators (focusing last domain performance), 3: use K+1 binary discriminators')
parser.add_argument('--dm_disc_coeff', default=0.1, type=float, help='Domain discriminator training coefficient')
parser.add_argument('--dm_disc_crit_weights', default=False, action='store_true', help='If True, set domain classifier criterion weights inversely proportional to # minibatches per each domain. If False, all the weights are set to 1')
parser.add_argument('--dm_adv_coeff', default=1, type=float, help='Domain-adversarial training coefficient')
parser.add_argument('--dm_adv_target_coeff', default=1, type=float, help='Multiply this additional coeff to dm_adv for the last domain')
parser.add_argument('--fix_dm_lambda', default=False, action='store_true', help='If False, increase the effect of auxiliary objectives from 0 to 1 w/ sigmoidal curve over the epochs. If True, fix the coefficient to be 1')
parser.add_argument('--dm_lambda_coeff', default=1, type=float, help='Coefficient to the inside expression of sigmoid')
parser.add_argument('--dm_adv_disc_l2reg', default=0.0000, type=float, help='L2 regularization coefficients for the discriminator. if --l2reg > 0, this is not used')
parser.add_argument('--dsn', default=False, action='store_true', help='Domain separation network objective')
parser.add_argument('--dsn_coeff', default=1, type=float, help='Domain separation coefficient')
parser.add_argument('--dsn_transp', default=False, action='store_true', help='If False, minimize cross covariance matrices. If True, minimize for soft subspace orthogonality')
parser.add_argument('--recon', default=False, action='store_true', help='Reconstruction objective')
parser.add_argument('--recon_coeff', default=0.1, type=float, help='Reconstruction coefficient')
parser.add_argument('--recon_use_utt', default=False, action='store_true', help='Add utterance encoder output as the input of the recon objective')
parser.add_argument('--src_dm_coeff', default=1, type=float, help='Weights to all the objectives for the source domains')
parser.add_argument('--src_dm_reverse_lambda', default=False, action='store_true', help='If True, multiply 1-lambda as the coefficient to the source domain objectives')
parser.add_argument('--tag_last_train_mb_cnt', default=-1, type=int, help='If -1, use all the trainset of the last domain for the tagging objective. Otherwise, set # mini-batches that will be used for the tag training objective.')
parser.add_argument('--all_last_train_mb_cnt', default=-1, type=int, help='If -1, use all the trainset of the last domain for all the objectives. Otherwise, set # mini-batches that will be used for all the training objectives.')
parser.add_argument('--tag_mid_train_mb_cnt', default=-1, type=int, help='If -1, use all the trainsets of the domains in the middle for the tagging objective. Otherwise, set # mini-batches that will be used for the tag training objective. Effective only when domainCnt >= 2')
parser.add_argument('--all_mid_train_mb_cnt', default=-1, type=int, help='If -1, use all the trainsets of the domains in the middle for all the objectives. Otherwise, set # mini-batches that will be used for all the training objectives. Effective only when domainCnt >= 2')
parser.add_argument('--tag_first_train_mb_cnt', default=-1, type=int, help='If -1, use all the trainset of the first domain for the tagging objective. Otherwise, set # mini-batches that will be used for the tag training objective. Effective only when domainCnt >= 2')
parser.add_argument('--all_first_train_mb_cnt', default=-1, type=int, help='If -1, use all the trainset of the first domain for all the objectives. Otherwise, set # mini-batches that will be used for all the training objectives. Effective only when domainCnt >= 2')
parser.add_argument('--max_mb_size', type=int, default=32, help='Maximum minibatch size')
parser.add_argument('--max_epoch', type=int, default=100, help='Maximum epoch')
parser.add_argument('--sgd_start_epoch', type=int, default=100, help='Change ADAM to SGD from this epoch')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
parser.add_argument('--l2reg', type=float, default=0, help='L2reg for all the used parameters')
parser.add_argument('--grad_norm_thrsd', type=float, default=5.0, help='Threshold for gradient normalization')
parser.add_argument('--seed', type=int, default=1000, help='Seed for random numbers')
parser.add_argument('--tag_acc_flag', default=False, action='store_true', help='If True, show accuracies rather than f1s')
parser.add_argument('--testout_file_prefix', type=str, default='', help='If a file name is specified, save the output of the testset with the best dev model to the given <prefix>_<domainIdx>.txt.')
parser.add_argument('--focus_last_domain', default=False, action='store_true', help='If true, pick the model showing the best devset score for the last domain. Otherwise, pick the model showing the best average devset score.')
parser.add_argument('--show_each_best_score', default=False, action='store_true', help='Show the best score for each domain at the end')
parser.add_argument('--no_cuda', default=False, action='store_true', help='Do not use CUDA')
parser.add_argument('--cudnn', default=False, action='store_true', help='Use CUDNN (cannot guarantee determinism e.g, when using CNN libraries)')
parser.add_argument('--logfile', type=str, default='', help='The file to save the logs. stdout is used if empty')
parser.add_argument('--utt_rep_save_prefix', type=str, default='', help='If not blank, save the utterance representations of the testset as a domain dictionalry of minibatch lists. The saved file names are <prefix>_<shared|sep|merged>_<domainIdx>.th')
args = parser.parse_args()

if args.no_shared_rnn == True or args.no_sep_rnns == True:
	if args.dsn == True:
		sys.stderr.print('--dsn is available when both shared rnn and sep rnns exist')
		sys.exit()
	if args.no_shared_rnn == True and args.no_sep_rnns == True:
		sys.stderr.print('At least one of shared rnn or sep rnn must exist')
		sys.exit()
	if args.no_shared_rnn == True and args.dm_adv >= 1:
		sys.stderr.print('--dm_adv is available only when shared rnn exist')
		sys.exit()

logfile = sys.stdout
if args.logfile != '':
	logfile = open(args.logfile, 'w')

random.seed(args.seed)
torch.manual_seed(args.seed)
if args.no_cuda == False:
	torch.cuda.manual_seed_all(args.seed)
	if args.cudnn == False:
		torch.backends.cudnn.enabled = False


# indices and corpora
wordIdxs, wordLists = [], []
charIdx, charList = {}, []
tagIdxs, tagLists = [], []
trainFileNames, devFileNames, testFileNames = [], [], []
langs = args.langs.split('_')
for lang in langs:
	curWordIdx, curWordList = util.loadIdx(args.corpus_dir + lang + '_word_idx.txt', True)	# reserve idx 0 for <pad>
	if args.recon == True:
		curWordIdx['<eos>'] = len(curWordIdx)
		curWordList.append('<eos>')
		curWordIdx['<bos>'] = len(curWordIdx)
		curWordList.append('<bos>')
	wordIdxs.append(curWordIdx)
	wordLists.append(curWordList)

	curCharIdx, curCharList = util.loadIdx(args.corpus_dir + lang + '_char_idx.txt')
	charList += curCharIdx

	curTagIdx, curTagList = util.loadIdx(args.corpus_dir + lang + '_tag_idx.txt')
	tagIdxs.append(curTagIdx)
	tagLists.append(curTagList)

	trainFileNames.append(args.corpus_dir + lang + '_train_shuffled.txt')
	devFileNames.append(args.corpus_dir + lang + '_dev.txt')
	testFileNames.append(args.corpus_dir + lang + '_test.txt')

domainCnt = len(trainFileNames)

if args.all_last_train_mb_cnt >= 0:
	trainFileNames[-1] = args.corpus_dir + langs[-1] + '_train_shuffled_' + str(args.max_mb_size * args.all_last_train_mb_cnt) + '.txt'
if domainCnt >= 3 and args.all_mid_train_mb_cnt >= 0:
	for i in range(1, domainCnt-1):
		trainFileNames[i] = args.corpus_dir + langs[i] + '_train_shuffled_' + str(args.max_mb_size * args.all_mid_train_mb_cnt) + '.txt'
if domainCnt >= 2 and args.all_first_train_mb_cnt >= 0:
	trainFileNames[0] = args.corpus_dir + langs[0] + '_train_shuffled_' + str(args.max_mb_size * args.all_first_train_mb_cnt) + '.txt'

charList = sorted(set(charList))
charList.insert(0, '<pad>')	# reserve idx 0 for <pad>
for i, c in enumerate(charList):
	charIdx[c] = i
#~


# models
wordEmbModels = []
for curWordIdx in wordIdxs:
	wordEmbModels.append(WordEmbModel(curWordIdx, args))
sharedModel = SharedModel(charIdx, domainCnt, args)
sepModels = []
for i in range(domainCnt):
	curSepModel = SepModel(len(wordIdxs[i]), tagIdxs[i])
	if args.no_sep_rnns == False and i > 0:
		curSepModel.sepRnn.load_state_dict(sepModels[0].sepRnn.state_dict())	# init the separate RNNs to have the same parameter values
		if args.rnn_prj == True:
			curSepModel.sepRnnPrj.load_state_dict(sepModels[0].sepRnnPrj.state_dict())
	sepModels.append(curSepModel)

tagCrit = nn.CrossEntropyLoss()
dmAdvCrits = []
dsnCrit = nn.L1Loss()
reconFwdCrit, reconBackCrit = nn.CrossEntropyLoss(), nn.CrossEntropyLoss()

if args.no_cuda == False:
	for curWordEmbModel in wordEmbModels:
		curWordEmbModel.cuda()
	sharedModel.cuda()
	for curSepModel in sepModels:
		curSepModel.cuda()
	tagCrit.cuda()
	dsnCrit.cuda()
	reconFwdCrit.cuda()
	reconBackCrit.cuda()

def setOptimMethod(optimFunc, initLr):
	sharedOpt = optimFunc(sharedModel.parameters(), initLr)
	sepOpts = []
	for i in range(domainCnt):
		sepOpts.append( optimFunc(list(wordEmbModels[i].parameters()) + list(sepModels[i].parameters()), initLr) )
	return sharedOpt, sepOpts

sharedOpt, sepOpts = setOptimMethod(optim.Adam, args.lr)
#~


class Utterance:	# represents an utterance
	def __init__(self, f):	# read an utterance from an opend file. If EOF arrived, the fields are returned empty
		self.words, self.tags = [], []
		self.maxWordLen = 0	# Length of the longest word

		dmPassed = False
		for line in f:
			tabs = line.split('\t')
			if len(tabs) == 1:
				token = tabs[0].strip()
				if len(token) == 0:
					return
				else:
					if args.intent == True:
						if dmPassed == False:
							self.domain = token
							dmPassed = True
						else:
							self.intent = token
			else:
				self.words.append(re.sub('[0-9]', '0', tabs[0]))
				self.maxWordLen = max(self.maxWordLen, len(tabs[0]))
				self.tags.append(tabs[1].strip())


	def empty(self):
		return len(self.words) == 0


class MiniBatch:
	wordBuf = torch.LongTensor()	# memory buffers for the inputs and the targets
	tagBuf = torch.LongTensor()
	padMaskBuf = torch.ByteTensor()
	charBuf = torch.LongTensor()
	wordIvtIdxBuf = torch.LongTensor()
	intentBuf = torch.LongTensor()
	domainBuf = torch.LongTensor() if args.dm_adv == 1 else torch.FloatTensor()
	reconFwdOutBuf, reconBackOutBuf = torch.LongTensor(), torch.LongTensor()

	wordTensor = torch.LongTensor()
	wordLenTensor = torch.LongTensor()
	tagTensor = torch.LongTensor()
	intentTensor = torch.LongTensor()
	charTensor = torch.LongTensor()
	wordIvtIdxTensor = torch.LongTensor()

	dsnBatch = Variable(torch.FloatTensor(1).zero_())


	def __init__(self, corpusFile, domainIdx, mbSize, evalFlag=True):
		utts = []
		for i in range(mbSize):
			curUtt = Utterance(corpusFile)
			if not curUtt.empty():
				utts.append(curUtt)
		maxSentLen = 0
		maxWordLen = 0
		if len(utts) > 0:
			utts = sorted(utts, key=lambda x: len(x.words), reverse=True)	# sort utterances in decreasing length order for dealing with paddings in RNNs
			self.sentLens = [len(x.words) for x in utts]
			maxSentLen = self.sentLens[0]
			self.padMask = torch.ByteTensor(maxSentLen, len(utts)).zero_()
			for i in range(len(self.sentLens)):
				self.padMask[0:self.sentLens[i], i].fill_(1)
			self.padMask = self.padMask.view(-1)
			MiniBatch.padMaskBuf.resize_(self.padMask.size()).copy_(self.padMask, True)
			self.padMask = Variable(MiniBatch.padMaskBuf, volatile=evalFlag)

			for utt in utts:
				maxWordLen = max(maxWordLen, utt.maxWordLen)
			maxWordLen += 2	# Include <w> and </w>

		MiniBatch.wordTensor.resize_(maxSentLen, len(utts)).fill_(wordIdxs[domainIdx]['<pad>'])
		MiniBatch.wordLenTensor.resize_(maxSentLen * len(utts)).fill_(2)
		MiniBatch.tagTensor.resize_(maxSentLen * len(utts))
		if args.intent == True:
			intentTensor.resize_(len(utts))

		for i, curUtt in enumerate(utts):
			for j in range(len(curUtt.words)):
				if curUtt.words[j] in wordIdxs[domainIdx]:
					MiniBatch.wordTensor[j, i] = wordIdxs[domainIdx][curUtt.words[j]]
				else:
					MiniBatch.wordTensor[j, i] = wordIdxs[domainIdx]['<unk>']

				MiniBatch.wordLenTensor[j*len(utts) + i] = len(curUtt.words[j]) + 2	# Include <w> and </w>

				if curUtt.tags[j] in tagIdxs[domainIdx]:
					MiniBatch.tagTensor[j*len(utts) + i] = tagIdxs[domainIdx][curUtt.tags[j]]
				else:
					sys.stderr.write('Unknown tag detected: ' + curUtt.tags[j] + '\n')
					sys.stderr.flush()
					sys.exit()
	
			if args.intent == True:		
				if curUtt.intent in intentIdxs[domainIdx]:
					intentTensor[i] = intentIdxs[domainIdx][curUtt.intent]
				else:
					intentTensor[i] = intentIdxs[domainIdx]['<unk>']

		MiniBatch.charTensor.resize_(maxWordLen, maxSentLen * len(utts)).fill_(charIdx['<pad>'])
		if len(utts) > 0:
			MiniBatch.wordLenTensor, wordSortIdxTensor = torch.sort(MiniBatch.wordLenTensor, dim=0, descending=True)
			self.wordLens = MiniBatch.wordLenTensor.tolist()
			ivtIdxSrc = torch.arange(0, wordSortIdxTensor.size(0)).long()
			MiniBatch.wordIvtIdxTensor.resize_as_(wordSortIdxTensor).scatter_(0, wordSortIdxTensor, ivtIdxSrc)

			MiniBatch.charTensor[0].fill_(charIdx['<w>'])
			MiniBatch.charTensor[1].fill_(charIdx['</w>'])
			for i, curUtt in enumerate(utts):
				for j, curWord in enumerate(curUtt.words):
					curCharPtr = MiniBatch.wordIvtIdxTensor[j*len(utts)+i]
					for k,c in enumerate(curWord):
						MiniBatch.charTensor[k+1, curCharPtr] = charIdx[c] if (c in charIdx) else charIdx['<unk>']
							
					MiniBatch.charTensor[ len(curWord)+1, curCharPtr ] = charIdx['</w>']


		MiniBatch.wordBuf.resize_(MiniBatch.wordTensor.size()).copy_(MiniBatch.wordTensor, True)
		self.wordBatch = Variable(MiniBatch.wordBuf, volatile=evalFlag)

		MiniBatch.charBuf.resize_(MiniBatch.charTensor.size()).copy_(MiniBatch.charTensor, True)
		MiniBatch.charBatch = Variable(MiniBatch.charBuf, volatile=evalFlag)

		MiniBatch.wordIvtIdxBuf.resize_(MiniBatch.wordIvtIdxTensor.size(0)).copy_(MiniBatch.wordIvtIdxTensor, True)
		MiniBatch.wordIvtIdxBatch = Variable(MiniBatch.wordIvtIdxBuf, volatile=evalFlag)

		MiniBatch.tagBuf.resize_(MiniBatch.tagTensor.size()).copy_(MiniBatch.tagTensor, True)
		self.tagBatch = Variable(MiniBatch.tagBuf, volatile=evalFlag)

		if args.intent == True:
			MiniBatch.intentBuf.resize_(intentTensor.size()).copy_(intentTensor, True)
			self.intentBatch = Variable(MiniBatch.intentBuf, volatile=evalFlag)

		if evalFlag == False:
			if args.dm_adv >= 1:
				dmTargetTensor = None
				if args.dm_adv == 1:
					dmTargetTensor = torch.LongTensor(len(utts)).fill_(domainIdx)
				else:
					if args.dm_adv == 2:
						dmTargetTensor = torch.FloatTensor(len(utts))
						if (domainIdx == domainCnt - 1):
							dmTargetTensor.fill_(1)
						else:
							dmTargetTensor.fill_(0)
					else:	#args.dm_adv == 3:
						dmTargetTensor = torch.FloatTensor(domainCnt, len(utts)).fill_(1)
						dmTargetTensor[domainIdx].fill_(0)

				MiniBatch.domainBuf.resize_(dmTargetTensor.size()).copy_(dmTargetTensor, True)
				self.domainBatch = Variable(MiniBatch.domainBuf, volatile=evalFlag)

			if args.recon == True:
				MiniBatch.reconFwdOutBuf.resize_(MiniBatch.wordTensor.size(0), MiniBatch.wordTensor.size(1))[:MiniBatch.wordTensor.size(0)-1].copy_(MiniBatch.wordTensor[1:], True)
				for i, sentLen in enumerate(self.sentLens):
					MiniBatch.reconFwdOutBuf[sentLen-1, i] = wordIdxs[domainIdx]['<eos>']
				self.reconFwdOutBatch = Variable(MiniBatch.reconFwdOutBuf, volatile=evalFlag)

				MiniBatch.reconBackOutBuf.resize_(MiniBatch.wordTensor.size(0), MiniBatch.wordTensor.size(1))[1:].copy_(MiniBatch.wordTensor[:MiniBatch.wordTensor.size(0)-1], True)
				for i, sentLen in enumerate(self.sentLens):
					MiniBatch.reconBackOutBuf[0, i] = wordIdxs[domainIdx]['<bos>']
				self.reconBackOutBatch = Variable(MiniBatch.reconBackOutBuf, volatile=evalFlag)


def trainMB(curDmMbIdx, curMB, domainIdx):
	sepOpts[domainIdx].zero_grad()
	wordEmbOut = wordEmbModels[domainIdx](curMB.wordBatch)

	SharedModel.setDomainIdx(domainIdx)
	sharedOpt.zero_grad()
	sharedOuts = sharedModel(curMB, wordEmbOut)

	sepOuts = sepModels[domainIdx](curDmMbIdx, curMB.sentLens, sharedOuts, sharedModel.sharedRnnDropout.noise if args.no_shared_rnn == False and args.rnn_dropout > 0 else None)
	
	loss = []

	dmAdvLossVals = [0] * domainCnt
	dmCorrectCnt, dmTotalCnt = 0, 0

	sharedOutIdx = 2
	sepOutIdx = 1
	if args.dm_adv >= 1 or args.recon == True:	# sharedUttEncOut
		sharedOutIdx += 1

	if args.dm_adv >= 1:	# Adversarial training
		dmAdvCnt = 1
		if (args.dm_adv == 2) and (domainIdx == domainCnt-1):
			dmAdvCnt = domainCnt-1
		elif args.dm_adv == 3:
			dmAdvCnt = domainCnt

		dmLabels = curMB.domainBatch
		for i in range(dmAdvCnt):
			if args.dm_adv == 1:
				dmCorrectCnt += (sharedOuts[sharedOutIdx].data.max(1)[1] == dmLabels.data).sum()
			else:
				if args.dm_adv == 3:
					dmLabels = dmLabels[i]
				dmCorrectCnt += ((sharedOuts[sharedOutIdx].data >= 0.5) == (dmLabels.data >= 0.5)).sum()
			dmTotalCnt += dmLabels.data.size(0)

			dmDiscCoeff = SharedModel.gradLambda * args.dm_disc_coeff
			if args.dm_disc_crit_weights == True:
				dmDiscCoeff *= sharedModel.mbDomainWeights[domainIdx]
			curDmAdvLoss = dmAdvCrits[i](sharedOuts[sharedOutIdx], dmLabels).mul_(dmDiscCoeff)
			dmAdvLossVals[domainIdx] += curDmAdvLoss.data[0]
			loss.append(curDmAdvLoss)
			sharedOutIdx += 1

		if args.dm_adv_disc_l2reg > 0 and args.l2reg == 0:
			for curClassifier in sharedModel.advDmClassifiers:
				for p in curClassifier.parameters():
					if p.grad is not None:
						p.grad.data.add_(args.dm_adv_disc_l2reg, p.data)

	dsnLossVal = 0	# Domain separation
	if args.dsn == True:
		dsnCoeff = args.dsn_coeff * SharedModel.gradLambda
		if args.dm_tag_weights == True:
			dsnCoeff *= sharedModel.mbDomainWeights[domainIdx]
		dsnLoss = dsnCrit(sepOuts[sepOutIdx], MiniBatch.dsnBatch).mul_(dsnCoeff)
		dsnLossVal = dsnLoss.data[0]
		loss.append(dsnLoss)
		sepOutIdx += 1

	reconLossVal = 0	# Reconstruction
	if args.recon == True:
		reconLossCoeff = args.recon_coeff * SharedModel.gradLambda
		if args.dm_tag_weights == True:
			reconLossCoeff *= sharedModel.mbDomainWeights[domainIdx]

		reconFwdOuts = torch.masked_select(sepOuts[sepOutIdx], curMB.padMask.unsqueeze(1).expand_as(sepOuts[sepOutIdx]))
		maskedReconFwdLabels = torch.masked_select(curMB.reconFwdOutBatch, curMB.padMask)
		reconLoss = reconFwdCrit(reconFwdOuts.view(-1, len(wordIdxs[domainIdx])), maskedReconFwdLabels).mul_(reconLossCoeff)
		sepOutIdx += 1

		reconBackOuts = torch.masked_select(sepOuts[sepOutIdx], curMB.padMask.unsqueeze(1).expand_as(sepOuts[sepOutIdx]))
		maskedReconBackLabels = torch.masked_select(curMB.reconBackOutBatch, curMB.padMask)
		reconLoss += reconBackCrit(reconBackOuts.view(-1, len(wordIdxs[domainIdx])), maskedReconBackLabels).mul_(reconLossCoeff)

		reconLossVal = reconLoss.data[0]
		loss.append(reconLoss)
		sepOutIdx += 1


	tagLossVal = 0	# Tagging
	if args.tag == True:
		if (domainCnt >= 2 and domainIdx == 0 and (args.tag_first_train_mb_cnt == -1 or curDmMbIdx < args.tag_first_train_mb_cnt)) or ((domainCnt >= 3 and domainIdx >= 1 and domainIdx < domainCnt-1) and (args.tag_mid_train_mb_cnt == -1 or curDmMbIdx < args.tag_mid_train_mb_cnt)) or (domainIdx == domainCnt-1 and (args.tag_last_train_mb_cnt == -1 or curDmMbIdx < args.tag_last_train_mb_cnt)):
			tagOuts = torch.masked_select(sepOuts[sepOutIdx], curMB.padMask.unsqueeze(1).expand_as(sepOuts[sepOutIdx]))
			maskedTags = torch.masked_select(curMB.tagBatch, curMB.padMask)
			tagLoss = tagCrit(tagOuts.view(-1, len(tagIdxs[domainIdx])), maskedTags)
			tagCoeff = 1
			if args.dm_tag_weights == True:
				tagCoeff *= sharedModel.mbTagWeights[domainIdx]
			if tagCoeff != 1:
				tagLoss.mul_(tagCoeff)
			tagLossVal = tagLoss.data[0]
			loss.append(tagLoss)

	if len(loss) == 0:
		return 0, [], 0, 0, 0, 0
	loss = sum(loss)	# combine all the losses
	if domainIdx < domainCnt-1:
		loss.mul_(args.src_dm_coeff * (1 - SharedModel.gradLambda if args.src_dm_reverse_lambda == True else 1))
	loss.backward()


	# l2reg
	if args.l2reg > 0:
		for p in wordEmbModels[domainIdx].parameters():
			p.grad.data.add_(args.l2reg, p.data)
		for p in sharedModel.parameters():
			if p.grad is not None:
				p.grad.data.add_(args.l2reg, p.data)
		for p in sepModels[domainIdx].parameters():
			if p.grad is not None:
				p.grad.data.add_(args.l2reg, p.data)
	#~

	if args.unk_word_zero == True:
		wordEmbModels[domainIdx].wordEmb.weight.grad.data[wordIdxs[domainIdx]['<unk>']].zero_()

	# gradient clipping
	normSum = 0
	for p in wordEmbModels[domainIdx].parameters():
		normSum += p.grad.data.norm() ** 2
	for p in sharedModel.parameters():
		if p.grad is not None:
			normSum += p.grad.data.norm() ** 2
	for p in sepModels[domainIdx].parameters():
		if p.grad is not None:
			normSum += p.grad.data.norm() ** 2
	normSum = math.sqrt(normSum)
	if normSum > args.grad_norm_thrsd:
		for p in wordEmbModels[domainIdx].parameters():
			p.grad.data.mul_(args.grad_norm_thrsd / normSum)
		for p in sharedModel.parameters():
			if p.grad is not None:
				p.grad.data.mul_(args.grad_norm_thrsd / normSum)
		for p in sepModels[domainIdx].parameters():
			if p.grad is not None:
				p.grad.data.mul_(args.grad_norm_thrsd / normSum)
	#~

	sepOpts[domainIdx].step()
	sharedOpt.step()
	return tagLossVal, dmAdvLossVals, dmCorrectCnt, dmTotalCnt, dsnLossVal, reconLossVal



def evalMB(curMB, domainIdx, testSetFlag, conllFile):
	wordEmbOut = wordEmbModels[domainIdx](curMB.wordBatch)
	SharedModel.setDomainIdx(domainIdx)
	sharedOuts = sharedModel(curMB, wordEmbOut, testSetFlag)
	sepOuts = sepModels[domainIdx](-1, curMB.sentLens, sharedOuts, sharedModel.sharedRnnDropout.noise if SharedModel.args.no_shared_rnn == False and SharedModel.args.rnn_dropout > 0 else None, testSetFlag)

	correctCnt, totalCnt = 0, 0
	if args.tag == True:
		maxIdxs = sepOuts[1].data.max(1)[1]
		maxIdxs = maxIdxs.view(curMB.wordBatch.data.size(0), -1)
		tagBatchView = curMB.tagBatch.data.view(curMB.wordBatch.data.size(0), -1)
		if args.tag_acc_flag == True:
			for i in range(maxIdxs.size(1)):
				for j in range(maxIdxs.size(0)):
					if curMB.wordBatch.data[j, i] != wordIdxs[domainIdx]['<pad>']:	# skip pads
						if tagBatchView[j, i] == maxIdxs[j, i]:
							correctCnt += 1
						totalCnt += 1
				
		if args.tag_acc_flag == False or (testSetFlag == True and args.testout_file_prefix != ''):
			curConllFile = conllFile
			for i in range(maxIdxs.size(1)):
				for j in range(maxIdxs.size(0)):
					if curMB.wordBatch.data[j, i] != wordIdxs[domainIdx]['<pad>']:	# skip pads
						curConllFile.write(' '.join([wordLists[domainIdx][curMB.wordBatch.data[j, i]], tagLists[domainIdx][tagBatchView[j, i]], tagLists[domainIdx][maxIdxs[j, i]]]))
						curConllFile.write('\n')
				curConllFile.write('\n')

	return correctCnt, totalCnt


def endConllF1(conllFile):
	conllFileName = conllFile.name
	conllFile.close()
	conllFile = open(conllFileName)
	out = subprocess.check_output('./conlleval.pl', stdin=conllFile)
	out = out.splitlines()
	fscore = float(out[1].split()[-1])
	conllFile.close()
	os.remove(conllFileName)
	return fscore


class EvalScore:
	def __init__(self, epoch, tagLoss, dmAdvLosses, dmCorrectCnts, dmTotalCnts, dsnLoss, reconLoss):
		self.epoch = epoch
		self.tagLoss = tagLoss
		self.dmAdvLosses = dmAdvLosses
		for i, t in enumerate(dmTotalCnts):
			if t != 0:
				dmCorrectCnts[i] /= t
		
		self.dmAccs = dmCorrectCnts
		self.devScores = [0] * len(devFileNames)
		self.testScores = [0] * len(testFileNames)
		self.avgDev = 0
		self.avgTest = 0
		self.dsnLoss = dsnLoss
		self.reconLoss = reconLoss

	def logResult(self):
		evalMetric = 'accs' if args.tag_acc_flag == True else 'f1s'
		self.avgDev = sum(self.devScores) / len(self.devScores)
		self.avgTest = sum(self.testScores) / len(self.testScores)
		logfile.write(', '.join(['epoch: ' + str(self.epoch+1), 'tag_loss: ' + str(self.tagLoss), evalMetric + ' avg_dev: ' + str(self.avgDev), 'avg_test: ' + str(self.avgTest), ' devs: ' + ', '.join([str(x) for x in self.devScores]), 'tests: ' + ', '.join([str(x) for x in self.testScores])]))
		if args.dm_adv >= 1:
			logfile.write(',\tadv_losses: ' + ', '.join([str(x) for x in self.dmAdvLosses]) + ', adv_accs train: ' + ', '.join([str(x) for x in self.dmAccs]))
		if args.dsn == True:
			logfile.write(',\tdsn_loss: ' + str(self.dsnLoss))
		if args.recon == True:
			logfile.write(',\trecon_loss: ' + str(self.reconLoss))


# setup for the main loop
logfile.write(str(args) + '\n')
logfile.flush()
random.seed(args.seed)	# initialize the random seed again so that the effects of different model parameters have less effects
torch.manual_seed(args.seed)
if args.no_cuda == False:
	torch.cuda.manual_seed_all(args.seed)
	MiniBatch.wordBuf, MiniBatch.tagBuf, MiniBatch.padMaskBuf, MiniBatch.charBuf, MiniBatch.wordIvtIdxBuf, MiniBatch.intentBuf, MiniBatch.domainBuf, MiniBatch.reconFwdOutBuf, MiniBatch.reconBackOutBuf, MiniBatch.dsnBatch = MiniBatch.wordBuf.cuda(), MiniBatch.tagBuf.cuda(), MiniBatch.padMaskBuf.cuda(), MiniBatch.charBuf.cuda(), MiniBatch.wordIvtIdxBuf.cuda(), MiniBatch.intentBuf.cuda(), MiniBatch.domainBuf.cuda(), MiniBatch.reconFwdOutBuf.cuda(), MiniBatch.reconBackOutBuf.cuda(), MiniBatch.dsnBatch.cuda()

startTime = time.time()
bestOverallEvalScore = EvalScore(0, 0, 0, [0]*domainCnt, [0]*domainCnt, 0, 0)
bestEvalScores = []
for i in range(domainCnt):
	bestEvalScores.append(EvalScore(0, 0, 0, [0]*domainCnt, [0]*domainCnt, 0, 0))

# domain minibatch shuffling
mbDomainIdxs = []
mbDomainWeights = torch.FloatTensor(domainCnt).zero_()	# weights for each domain
for i,trainFile in enumerate(trainFileNames):
	trainFile = open(trainFile)
	while True:
		curMB = MiniBatch(trainFile, i, args.max_mb_size)
		if curMB.wordBatch.data.dim() != 0 :
			mbDomainIdxs.append(i)
			mbDomainWeights[i] += 1
		else:
			break

if args.dm_tag_weights == True or args.dm_disc_crit_weights == True:
	mbDomainCntSum = mbDomainWeights.sum()
	for i in range(domainCnt):
		sharedModel.mbDomainWeights[i] = (1 - mbDomainWeights[i] / mbDomainCntSum) / (domainCnt-1)	# Set the domain weight proportional to the number of minibatches per each domain
	if args.pivot_weights == True:
		for i in range(domainCnt):
			sharedModel.mbDomainWeights[i] /= sharedModel.mbDomainWeights[domainCnt-1]	# Let the target weight to be always 1

	if args.tag_last_train_mb_cnt >= 1:
		mbDomainWeights[domainCnt-1] = args.tag_last_train_mb_cnt
	if domainCnt >= 3 and args.tag_mid_train_mb_cnt >= 1:
		for i in range(1, domainCnt-1):
			mbDomainWeights[i] = args.tag_mid_train_mb_cnt
	if domainCnt >= 2 and args.tag_first_train_mb_cnt >= 1:
		mbDomainWeights[0] = args.tag_first_train_mb_cnt
	mbDomainCntSum = mbDomainWeights.sum()
	for i in range(domainCnt):
		sharedModel.mbTagWeights[i] = (1 - mbDomainWeights[i] / mbDomainCntSum) / (domainCnt-1)
	if args.pivot_weights == True:
		for i in range(domainCnt):
			sharedModel.mbTagWeights[i] /= sharedModel.mbTagWeights[domainCnt-1]


if args.dm_adv >= 1:	# adversarial training setting
	if args.dm_adv == 1:
		dmAdvCrits.append( nn.CrossEntropyLoss() )
	else:
		for i in range(domainCnt-1 if (args.dm_adv == 2) else domainCnt):
			dmAdvCrits.append( nn.BCELoss() )

	if args.no_cuda == False:
		for curCrit in dmAdvCrits:
			curCrit.cuda()
#~

# main loop
for epoch in range(args.max_epoch):
	# train
	if epoch == args.sgd_start_epoch:
		sharedOpt, sepOpts = setOptimMethod(optim.SGD, 0.5)
		logfile.write('Changed ADAM to SGD\n')

	if args.fix_dm_lambda == False:
		SharedModel.gradLambda = 2 * (nn.Sigmoid()( Variable(torch.FloatTensor(1).fill_(10.0 * args.dm_lambda_coeff * epoch / (args.max_epoch-1)), volatile=True) )).data[0] - 1	# increase the effect of auxiliary objectives from 0 to 1 within a sigmoidal curve

	tagLossSum, dmAdvLossSums, dmCorrectCnts, dmTotalCnts, dsnLossSum, reconLossSum = 0, [0]*domainCnt, [0]*domainCnt, [0]*domainCnt, 0, 0
	random.shuffle(mbDomainIdxs)
	for curWordEmbModel in wordEmbModels:
		curWordEmbModel.train()
	sharedModel.train()
	for curSepModel in sepModels:
		curSepModel.train()
	trainFiles = [open(x) for x in trainFileNames]
	dmMbIdxs = [0]*domainCnt	# minibatch index for each domain
	for domainIdx in mbDomainIdxs:
		curMB = MiniBatch(trainFiles[domainIdx], domainIdx, args.max_mb_size, False)
		if curMB.wordBatch.data.dim() != 0:
			mbTagLoss, mbDmAdvLosses, mbDmCorrectCnt, mbDmTotalCnt, mbDsnLoss, mbReconLoss = trainMB(dmMbIdxs[domainIdx], curMB, domainIdx)
			tagLossSum += mbTagLoss
			for i,l in enumerate(mbDmAdvLosses):
				dmAdvLossSums[i] += l
			dmCorrectCnts[domainIdx] += mbDmCorrectCnt
			dmTotalCnts[domainIdx] += mbDmTotalCnt
			dsnLossSum += mbDsnLoss
			reconLossSum += mbReconLoss

		dmMbIdxs[domainIdx] += 1

	for trainFile in trainFiles:
		trainFile.close()
	#~


	for curWordEmbModel in wordEmbModels:
		curWordEmbModel.eval()
	sharedModel.eval()
	for curSepModel in sepModels:
		curSepModel.eval()
	curScore = EvalScore(epoch, tagLossSum, dmAdvLossSums, dmCorrectCnts, dmTotalCnts, dsnLossSum, reconLossSum)

	# dev
	for domainIdx in range(len(devFileNames)):
		devCorrectCnt, devTotalCnt, devScore = 0, 0, 0
		evalFile = open(devFileNames[domainIdx])
		conllFile = tempfile.NamedTemporaryFile(mode='w', delete=False) if args.tag_acc_flag == False else None
		while True:
			curMB = MiniBatch(evalFile, domainIdx, args.max_mb_size)
			if curMB.wordBatch.data.dim() != 0:
				curCorrectCnt, curTotalCnt = evalMB(curMB, domainIdx, False, conllFile)
				devCorrectCnt += curCorrectCnt
				devTotalCnt += curTotalCnt
			else:
				if args.tag_acc_flag == True:
					devScore = devCorrectCnt / devTotalCnt
				else:
					devScore = endConllF1(conllFile)
				curScore.devScores[domainIdx] = devScore
				evalFile.close()
				break
	#~

	# test
	for domainIdx in range(len(testFileNames)):
		sepModels[domainIdx].uttSharedReps = []
		sepModels[domainIdx].uttMergedReps = []
		testCorrectCnt, testTotalCnt, testScore = 0, 0, 0
		evalFile = open(testFileNames[domainIdx])
		conllFile = tempfile.NamedTemporaryFile(mode='w', delete=False) if args.tag_acc_flag == False or args.testout_file_prefix != '' else None
		while True:
			curMB = MiniBatch(evalFile, domainIdx, args.max_mb_size)
			if curMB.wordBatch.data.dim() != 0:
				curCorrectCnt, curTotalCnt = evalMB(curMB, domainIdx, True, conllFile)
				testCorrectCnt += curCorrectCnt
				testTotalCnt += curTotalCnt
			else:
				if args.tag_acc_flag == True:
					testScore = testCorrectCnt / testTotalCnt
					if args.testout_file_prefix != '':
						if curScore.devScores[domainIdx] > bestEvalScores[domainIdx].devScores[domainIdx] or (curScore.devScores[domainIdx] == bestEvalScores[domainIdx].devScores[domainIdx] and curScore.avgDev > bestEvalScores[domainIdx].avgDev):
							tempTestFileName = conllFile.name
							conllFile.close()
							shutil.copyfile(tempTestFileName, args.testout_file_prefix + '_' + str(domainIdx) + '.txt')	# Save the test output for the best dev model
						endConllF1(conllFile)
				else:
					testScore = endConllF1(conllFile)
				curScore.testScores[domainIdx] = testScore
				evalFile.close()
				break
	#~

	# report
	curScore.logResult()

	for i in range(domainCnt):
		if curScore.devScores[i] > bestEvalScores[i].devScores[i] or curScore.devScores[i] == bestEvalScores[i].devScores[i] and curScore.avgDev > bestEvalScores[i].avgDev:
			bestEvalScores[i] = curScore

	bestDevFlag = False
	if args.focus_last_domain == True and curScore.devScores[-1] > bestOverallEvalScore.devScores[-1] or curScore.devScores[-1] == bestOverallEvalScore.devScores[-1] and curScore.avgDev > bestOverallEvalScore.avgDev:
		bestDevFlag = True
	elif args.focus_last_domain == False and curScore.avgDev > bestOverallEvalScore.avgDev:
		bestDevFlag = True

	if bestDevFlag == True:
		bestOverallEvalScore = curScore
		logfile.write('\t\tbest ' + ('last' if args.focus_last_domain == True else 'avg') + ' dev so far')
		if len(args.utt_rep_save_prefix) > 0:
			for i in range(domainCnt):
				torch.save(sepModels[i].uttSharedReps, args.utt_rep_save_prefix + '_shared_' + str(i) + '.th')
				if args.no_sep_rnns == False:
					torch.save(sepModels[i].uttMergedReps, args.utt_rep_save_prefix + '_merged_' + str(i) + '.th')

	logfile.write('\n')
	logfile.flush()

if args.show_each_best_score == True:
	logfile.write('best epochs: ')
	for i in range(domainCnt):
		logfile.write(str(bestEvalScores[i].epoch) + ', ')
	logfile.write('\tdevs: ')
	for i in range(domainCnt):
		logfile.write(str(bestEvalScores[i].devScores[i]) + ', ')
	logfile.write('\ttests: ')
	for i in range(domainCnt):
		logfile.write(str(bestEvalScores[i].testScores[i]) + ', ')
	logfile.write('\n')
logfile.write('best ' + ('last' if args.focus_last_domain == True else 'avg') + ': ')
bestOverallEvalScore.logResult()
logfile.write('\t\t' + str(time.time()-startTime) + ' seconds elapsed.\n')
