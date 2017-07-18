# -------------------------------------------------------
# Joo-Kyung Kim (kimjook@cse.ohio-state.edu)	7/15/2017
# -------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.utils.rnn as rnn_utils
import torch.nn.init

import util


def getAttnOutput(input, attnScorer, winSize=0):	# get attention output following [Liu and Lane, Interspeech 2016]. the input is seqlen X batchsize X dim. if winSize is 0, all the time steps are used for the weigted averaging
	attnSeq = []
	for i in range(input.size(0)):
		curSeq = []
		if i > 0:
			leftBegin = 0
			if winSize > 0:
				leftBegin = max(0, i-winSize)
			curSeq.append(input[leftBegin:i])
		if i < input.size(0):
			leftEnd = input.size(0)
			if winSize > 0:
				leftEnd = min(i+winSize+1, input.size(0))
			curSeq.append(input[i:leftEnd])
		curSeq = torch.cat(curSeq, 0)
		cur = input[i:i+1].expand_as(curSeq)

		attnScores = attnScorer( torch.cat([cur, curSeq], 2).view(-1, 2*input.size(2)) )	# get attention scores
		transAttnScores = attnScores.view(curSeq.size(0), input.size(1)).transpose(0, 1)	# batchSize X curSeqLen
		smOut = F.softmax(transAttnScores).transpose(0, 1)
		smOutSeq = smOut.unsqueeze(2).expand_as(curSeq)
		weightedAvgSeq = (curSeq * smOutSeq).sum(0)
		attnSeq.append(weightedAvgSeq)
	attnSeq = torch.cat(attnSeq, 0)
	return torch.cat([input, attnSeq], 2)


def setUttEncoder(module):	# set utterance encoder to the module
	if SharedModel.args.utt_enc_noise == True:
		module.uttEncNoise = Variable(torch.FloatTensor(), volatile=True)
		if SharedModel.args.no_cuda == False:
			module.uttEncNoise = module.uttEncNoise.cuda()

	if SharedModel.args.utt_enc_type >= 2:
		module.uttEncoder = nn.ModuleList()
		for i in [int(x) for x in SharedModel.args.conv_filters.split('_')]:
			module.uttEncoder.append( nn.Conv1d(2*SharedModel.args.hid_dim * (2 if SharedModel.args.attn == 2 else 1), SharedModel.args.conv_out_dim, i, 1, int(math.ceil((i-1)/2))) )

	if SharedModel.args.utt_enc_bn == True:
		uttEncOutSize = 2 * SharedModel.args.hid_dim
		if SharedModel.args.utt_enc_type >= 2:
			uttEncOutSize = 3 * SharedModel.args.conv_out_dim
		elif SharedModel.args.attn == 2:
			uttEncOutSize = 4 * SharedModel.args.hid_dim
		module.uttBn = nn.BatchNorm1d(uttEncOutSize)


def fwdUttEnc(module, rnnOut):	# forward utterance encoder to get the utterance representation
	uttEncOut = None
	if SharedModel.args.utt_enc_type == 0:	# Encoding by summation
		uttEncOut = rnnOut.sum(0).squeeze(0)
	elif SharedModel.args.utt_enc_type == 1:	# Encoding by mean
		uttEncOut = rnnOut.mean(0).squeeze(0)
	else:	# Encoding by CNN
		uttEncOut = []
		for i, curConv in enumerate(module.uttEncoder):
			curConvInput = rnnOut.permute(1, 2, 0)
			curConvOut = curConv(curConvInput)
			curPoolOut = None
			if SharedModel.args.utt_enc_type == 2:	# using average pooling
				curPoolOut = F.avg_pool1d(curConvOut, curConvOut.data.size(2))
			else:	# using max pooling
				curPoolOut = F.max_pool1d(curConvOut, curConvOut.data.size(2))
			uttEncOut.append(curPoolOut)
		uttEncOut = torch.cat(uttEncOut, 1)
		uttEncOut = uttEncOut.squeeze(2)
		uttEncOut = F.tanh(uttEncOut)

	if SharedModel.args.utt_enc_noise == True:
		module.uttEncNoise.data.resize_(uttEncOut.size()).normal_(0, 0.1)	# Add white noises to the utterance encoding
		uttEncOut.add_(module.uttEncNoise)

	if SharedModel.args.utt_enc_bn == True:
		uttEncOut = module.uttBn(uttEncOut)
	return uttEncOut




class WordEmbModel(nn.Module):
	def __init__(self, wordIdx, args):
		super(WordEmbModel, self).__init__()

		self.wordEmb = nn.Embedding(len(wordIdx), args.emb_dim, padding_idx=wordIdx['<pad>'])	# idx 0 for <pad>
		if args.init_word_vec != '':
			self.wordEmb.weight.data = torch.load(args.init_word_vec)	# should fix to be able to deal with <eos>
		else:
			torch.nn.init.xavier_normal(self.wordEmb.weight.data)	# Xavier initiaization to word vectors

		self.wordEmb.weight.data[wordIdx['<pad>']].zero_()	# <pad>
		if args.unk_word_zero == True:
			self.wordEmb.weight.data[wordIdx['<unk>']].zero_()


	def forward(self, wordBatch):
		return self.wordEmb(wordBatch)




class SharedModel(nn.Module):
	args = None
	gradLambda = 1	# coefficients for auxiliary gradient modification
	domainIdx = None
	domainCnt = 0


	@staticmethod
	def setDomainIdx(domainIdx):
		SharedModel.domainIdx = domainIdx
	

	def __init__(self, charIdx, domainCnt, args):
		super(SharedModel, self).__init__()

		SharedModel.args = args
		SharedModel.domainCnt = domainCnt
		self.mbDomainWeights = torch.FloatTensor(domainCnt).fill_(1)
		self.mbTagWeights = torch.FloatTensor(domainCnt).fill_(1)

		# shared charEmb
		self.charEmb = nn.Embedding(len(charIdx), args.char_emb_dim, padding_idx=charIdx['<pad>'])	# idx 0 for <pad>
		torch.nn.init.xavier_normal(self.charEmb.weight.data)
		self.charEmb.weight.data[charIdx['<pad>']].zero_()
		self.charEmb.weight.data[charIdx['<unk>']].zero_()

		self.charH0c0 = [torch.FloatTensor(), torch.FloatTensor()]
		if args.no_cuda == False:
			self.charH0c0[0], self.charH0c0[1] = self.charH0c0[0].cuda(), self.charH0c0[1].cuda()

		self.charRnn = nn.LSTM(args.char_emb_dim, args.char_hid_dim, 1, True, False, 0, True)
		util.initLSTM(self.charRnn, 1, True)
		#~

		if args.word_dropout > 0:
			self.wordDropout = util.SeqConstDropout(args.word_dropout, True)

		if args.no_shared_rnn == False:	# shared RNN
			self.h0c0 = [torch.FloatTensor(), torch.FloatTensor()]
			if args.no_cuda == False:
				self.h0c0[0], self.h0c0[1] = self.h0c0[0].cuda(), self.h0c0[1].cuda()

			self.sharedRnn = nn.LSTM(args.emb_dim+2*args.char_hid_dim, args.hid_dim, args.rnn_layer_cnt, True, False, 0, True)
			util.initLSTM(self.sharedRnn, args.rnn_layer_cnt, True)

			if args.rnn_dropout > 0:
				self.sharedRnnDropout = util.SeqConstDropout(args.rnn_dropout, True)

			if args.rnn_prj == True:
				self.sharedRnnPrj = nn.Linear(2*args.hid_dim, 2*args.hid_dim)
			#~

			if args.attn == 2:	# separate attentions
				self.sharedAttnScorer = nn.Linear(4 * args.hid_dim, 1)

			if args.dm_adv >= 1 or (args.recon == True and args.recon_use_utt == True):
				setUttEncoder(self)	# Set utterance encoder
				self.uttEncDim = 2*args.hid_dim if args.utt_enc_type <= 1 else 3*args.conv_out_dim

			# Domain-adversarial training on the shared RNN outputs
			if args.dm_adv >= 1:
				def grad_reverse_func(module, grad_i, grad_o):
					gradAdvCoeff = - SharedModel.args.dm_adv_coeff
					if SharedModel.domainIdx == domainCnt-1:	# when the current domain is the target (last domain)
						gradAdvCoeff *= args.dm_adv_target_coeff
					return gradAdvCoeff*grad_i[0], grad_i[1], grad_i[2]

				def getDmClassifier(targetCnt):
					curClassifier = nn.Sequential(nn.Linear(self.uttEncDim, args.hid_dim), nn.LeakyReLU(0.2, True), nn.Linear(args.hid_dim, targetCnt))
					curClassifier[len(curClassifier._modules)-1].bias.data.zero_()
					curClassifier[0].register_backward_hook(grad_reverse_func)	# reverse the sign of the input gradients for adversarial training
					return curClassifier

				self.advDmClassifiers = nn.ModuleList()
				if args.dm_adv == 1:
					self.advDmClassifiers.append(getDmClassifier(domainCnt))
				else:
					for i in range(domainCnt-1 if (args.dm_adv == 2) else domainCnt):
						self.advDmClassifiers.append(getDmClassifier(1))
			#~



	def forward(self, curMB, wordOut, testSetFlag=False):
		# char emb and rnn
		charOut = self.charEmb(curMB.charBatch)
		charOut = rnn_utils.pack_padded_sequence(charOut, curMB.wordLens)
		curCharHC = (Variable(self.charH0c0[0].resize_(2, len(curMB.wordLens), SharedModel.args.char_hid_dim).zero_()), Variable(self.charH0c0[1].resize_(2, len(curMB.wordLens), SharedModel.args.char_hid_dim).zero_()))
		charRnnOut, curCharHC = self.charRnn(charOut, curCharHC)
		charRnnOut, charRnnOutLens = rnn_utils.pad_packed_sequence(charRnnOut)
		del charRnnOut
		lastCharHidOuts = curCharHC[0].transpose(0,1)
		lastCharHidOuts = lastCharHidOuts.contiguous().view(lastCharHidOuts.size(0), -1)
		lastCharHidOuts = lastCharHidOuts.index_select(0, curMB.wordIvtIdxBatch)
		#~

		# Concated word vectors
		wordOut = torch.cat([wordOut, lastCharHidOuts.view(wordOut.size(0), wordOut.size(1), -1)], 2)
		if SharedModel.args.word_dropout > 0:
			wordOut = self.wordDropout(wordOut)

		packedWordOut = rnn_utils.pack_padded_sequence(wordOut, curMB.sentLens)
		#~

		finalOut = [packedWordOut]
		if SharedModel.args.no_shared_rnn == True:
			return finalOut

		# shared RNN
		curHC = (Variable(self.h0c0[0].resize_(2*SharedModel.args.rnn_layer_cnt, len(curMB.sentLens), SharedModel.args.hid_dim).zero_()), Variable(self.h0c0[1].resize_(2*SharedModel.args.rnn_layer_cnt, len(curMB.sentLens), SharedModel.args.hid_dim).zero_()))
		sharedRnnOut, curHC = self.sharedRnn(packedWordOut, curHC)
		sharedRnnOut, sharedRnnOutLens = rnn_utils.pad_packed_sequence(sharedRnnOut)

		if SharedModel.args.rnn_dropout > 0:
			sharedRnnOut = self.sharedRnnDropout(sharedRnnOut)

		if SharedModel.args.rnn_prj == True:
			sharedRnnOut = self.sharedRnnPrj(sharedRnnOut.view(sharedRnnOut.size(0)*sharedRnnOut.size(1), sharedRnnOut.size(2))).view(sharedRnnOut.size(0), sharedRnnOut.size(1), sharedRnnOut.size(2))

		if SharedModel.args.attn == 2:	# separate attentions
			sharedRnnOut = getAttnOutput(sharedRnnOut, self.sharedAttnScorer, SharedModel.args.attn_winsize)
		#~
		finalOut.append(sharedRnnOut)

		uttEncOut = None
		if (self.training == True and (SharedModel.args.dm_adv >= 1 or (SharedModel.args.recon == True and SharedModel.args.recon_use_utt == True))) or (testSetFlag == True and len(SharedModel.args.utt_rep_save_prefix) > 0):	# Encoded utterance representation
			uttEncOut = fwdUttEnc(self, sharedRnnOut)
			finalOut.append(uttEncOut)

		if self.training == True:
			# domain-adversarial training
			if SharedModel.args.dm_adv >= 1:
				# domain discriminator
				if SharedModel.args.dm_adv == 1:
					finalOut.append( self.advDmClassifiers[0](uttEncOut) )	# CrossEntropyLoss is used for this case. BCELoss is used for others
				elif SharedModel.args.dm_adv == 2:
					if SharedModel.domainIdx < SharedModel.domainCnt-1:
						finalOut.append( F.sigmoid( self.advDmClassifiers[SharedModel.domainIdx](uttEncOut) ) )
					else:
						for i in range(SharedModel.domainCnt-1):
							finalOut.append( F.sigmoid( self.advDmClassifiers[i](uttEncOut) ) )
				elif SharedModel.args.dm_adv == 3:
					for i in range(SharedModel.domainCnt):
						finalOut.append( F.sigmoid( self.advDmClassifiers[i](uttEncOut) ) )
				#~
			#~

		return finalOut



class SepModel(nn.Module):
	@staticmethod
	def _dsnHookFunc(grad):
		return (SharedModel.gradLambda * SharedModel.args.dsn_coeff) * grad

	
	def __init__(self, lenWordIdx, tagIdx):
		super(SepModel, self).__init__()

		# sep RNN
		if SharedModel.args.no_sep_rnns == False:	# using domain-specific RNN
			self.h0c0 = [torch.FloatTensor(), torch.FloatTensor()]	# RNN init hidden and cell states
			if SharedModel.args.no_cuda == False:
				self.h0c0[0], self.h0c0[1] = self.h0c0[0].cuda(), self.h0c0[1].cuda()

			self.sepRnn = nn.LSTM(SharedModel.args.emb_dim+2*SharedModel.args.char_hid_dim, SharedModel.args.hid_dim, SharedModel.args.rnn_layer_cnt, True, False, 0, True)
			util.initLSTM(self.sepRnn, SharedModel.args.rnn_layer_cnt, True)

			if SharedModel.args.rnn_dropout > 0:
				self.sepRnnDropout = util.SeqConstDropout(SharedModel.args.rnn_dropout, True, True if SharedModel.args.no_shared_rnn == False else False)	# lazy noise generation so that shared and sep have the same noise mask

			if SharedModel.args.rnn_prj == True:
				self.sepRnnPrj = nn.Linear(2*SharedModel.args.hid_dim, 2*SharedModel.args.hid_dim)
		#

		self.uttSharedReps, self.uttMergedReps = [], []	# keep utterance representation minibatch for saving to a file

		if SharedModel.args.attn >= 1:	# attention
			attnDim = 4 * SharedModel.args.hid_dim
			if SharedModel.args.attn == 1 and SharedModel.args.rnn_merge_cat == True:
				attnDim *= 2
			self.sepAttnScorer = nn.Linear(attnDim, 1)

		self.tagInputDim = SharedModel.args.hid_dim
		if SharedModel.args.rnn_merge_cat == False and SharedModel.args.attn == 0:
			self.tagInputDim *= 2
		elif SharedModel.args.rnn_merge_cat == True and SharedModel.args.attn >= 1:
			self.tagInputDim *= 8
		else:
			self.tagInputDim *= 4

		if SharedModel.args.recon == True:
			if SharedModel.args.recon_use_utt == True:
				setUttEncoder(self)	# Set utterance encoder
				self.uttEncDim = 2*SharedModel.args.hid_dim if SharedModel.args.utt_enc_type <= 1 else 3*SharedModel.args.conv_out_dim
				self.uttEncPrjLayer = nn.Linear(self.uttEncDim, SharedModel.args.hid_dim)

			self.reconFwdPrjLayer = nn.Linear(SharedModel.args.hid_dim * (1 if SharedModel.args.rnn_merge_cat == False else 2), SharedModel.args.hid_dim)
			self.reconFwdOutLayer = nn.Sequential(nn.Tanh(), nn.Linear(SharedModel.args.hid_dim * (2 if SharedModel.args.recon_use_utt == True else 1), lenWordIdx))
			self.reconFwdOutLayer[len(self.reconFwdOutLayer._modules)-1].bias.data.zero_()

			self.reconBackPrjLayer = nn.Linear(SharedModel.args.hid_dim * (1 if SharedModel.args.rnn_merge_cat == False else 2), SharedModel.args.hid_dim)
			self.reconBackOutLayer = nn.Sequential(nn.Tanh(), nn.Linear(SharedModel.args.hid_dim * (2 if SharedModel.args.recon_use_utt == True else 1), lenWordIdx))
			self.reconBackOutLayer[len(self.reconBackOutLayer._modules)-1].bias.data.zero_()

		if SharedModel.args.tag == True:	# tagging
			self.tagOutLayer = nn.Linear(self.tagInputDim, len(tagIdx))
			self.tagOutLayer.bias.data.zero_()



	def forward(self, curDmMbIdx, sentLens, sharedOuts, sharedRnnDropoutNoise, testSetFlag=False):
		packedWordOut = sharedOuts[0]
		sharedRnnOut = sharedOuts[1] if SharedModel.args.no_shared_rnn == False else None
		sepRnnOut, mergedRnnOut = None, None

		if SharedModel.args.no_sep_rnns == False:
			sepHC = (Variable(self.h0c0[0].resize_(2*SharedModel.args.rnn_layer_cnt, len(sentLens), SharedModel.args.hid_dim).zero_()), Variable(self.h0c0[1].resize_(2*SharedModel.args.rnn_layer_cnt, len(sentLens), SharedModel.args.hid_dim).zero_()))
			sepRnnOut, sepHC = self.sepRnn(packedWordOut, sepHC)
			sepRnnOut, sepRnnOutLens = rnn_utils.pad_packed_sequence(sepRnnOut)

			if SharedModel.args.rnn_dropout > 0:
				if SharedModel.args.no_shared_rnn == False:
					self.sepRnnDropout.noise = sharedRnnDropoutNoise
				sepRnnOut = self.sepRnnDropout(sepRnnOut)

			if SharedModel.args.rnn_prj == True:
				sepRnnOut = self.sepRnnPrj(sepRnnOut.view(sepRnnOut.size(0)*sepRnnOut.size(1), sepRnnOut.size(2))).view(sepRnnOut.size(0), sepRnnOut.size(1), sepRnnOut.size(2))

			if SharedModel.args.attn == 2:	# separate attentions
				sepRnnOut = getAttnOutput(sepRnnOut, self.sepAttnScorer, SharedModel.args.attn_winsize)

		
		finalOut = []

		if SharedModel.args.no_shared_rnn == False and SharedModel.args.no_sep_rnns == False and SharedModel.args.no_sep_for_pred == False:	# mergedRnnOut is necessary when training with labels of the current domain
			if SharedModel.args.rnn_merge_cat == True:
				mergedRnnOut = torch.cat([sharedRnnOut, sepRnnOut], 2)
			else:
				mergedRnnOut = (sharedRnnOut + sepRnnOut)
		
			if SharedModel.args.attn == 1:	# attention after merging
				mergedRnnOut = getAttnOutput(mergedRnnOut, self.sepAttnScorer, SharedModel.args.attn_winsize)
		elif SharedModel.args.no_shared_rnn == False:
			mergedRnnOut = sharedRnnOut
		else:
			mergedRnnOut = sepRnnOut

		finalOut.append(mergedRnnOut)
		

		sharedUttEncOut, sepUttEncOut, mergedUttEncOut = None, None, None
		if (self.training == True and (SharedModel.args.recon == True and SharedModel.args.recon_use_utt)) or (testSetFlag == True and len(SharedModel.args.utt_rep_save_prefix) > 0):	# Encoded utterance representation
			if SharedModel.args.no_shared_rnn == False:
				sharedUttEncOut = sharedOuts[2]
				mergedUttEncOut = sharedUttEncOut
			if SharedModel.args.no_sep_rnns == False:
				sepUttEncOut = fwdUttEnc(self, sepRnnOut)
				if mergedUttEncOut is not None:
					mergedUttEncOut = mergedUttEncOut + sepUttEncOut
				else:
					mergedUttEncOut = sepUttEncOut

			if testSetFlag == True and len(SharedModel.args.utt_rep_save_prefix) > 0:	# save RNN outputs for analyses
				self.uttSharedReps.append(sharedUttEncOut)
				if SharedModel.args.no_sep_rnns == False:
					self.uttMergedReps.append(mergedUttEncOut)


		if self.training == True:
			# domain separation network
			if SharedModel.args.dsn == True:
				zeroMeanUnitNormSharedRnnOut = sharedRnnOut - sharedRnnOut.detach().mean(2).expand_as(sharedRnnOut)
				zeroMeanUnitNormSharedRnnOut.register_hook(self._dsnHookFunc)
				normSharedRnnOut = zeroMeanUnitNormSharedRnnOut.detach().norm(2,2).add_(1e-10).expand_as(sharedRnnOut)
				zeroMeanUnitNormSharedRnnOut = zeroMeanUnitNormSharedRnnOut.div(normSharedRnnOut)

				zeroMeanUnitNormSepRnnOut = sepRnnOut - sepRnnOut.detach().mean(2).expand_as(sepRnnOut)
				zeroMeanUnitNormSepRnnOut.register_hook(self._dsnHookFunc)
				normSepRnnOut = zeroMeanUnitNormSepRnnOut.detach().norm(2,2).add_(1e-10).expand_as(sepRnnOut)
				zeroMeanUnitNormSepRnnOut = zeroMeanUnitNormSepRnnOut.div(normSepRnnOut)

				dsnMat = None
				if SharedModel.args.dsn_transp == False:
					dsnMat = zeroMeanUnitNormSharedRnnOut.transpose(1,2).bmm( zeroMeanUnitNormSepRnnOut )	# reduce cross-covaraince
				else:
					dsnMat = zeroMeanUnitNormSharedRnnOut.bmm( zeroMeanUnitNormSepRnnOut.transpose(1,2) )	# encourage soft subspace orthogonality

				dsnOut = dsnMat.add_(1e-10).norm().div_(sharedRnnOut.size(0)*sharedRnnOut.size(1))
				finalOut.append(dsnOut)
			#~

			# reconstruction (bidirectional language modeling)
			if SharedModel.args.recon == True:
				mergedRnnOutView = mergedRnnOut.view(mergedRnnOut.size(0) * mergedRnnOut.size(1), mergedRnnOut.size(2))
				mergedVecLen = mergedRnnOut.size(2)
				mergedRnnOutFwdView, mergedRnnOutFwdView = None, None

				if SharedModel.args.rnn_merge_cat == False:
					if SharedModel.args.attn == 0:
						mergedRnnOutFwdView = mergedRnnOutView[:, :mergedVecLen//2]
						mergedRnnOutBackView = mergedRnnOutView[:, mergedVecLen//2:]
					else:
						mergedRnnOutFwdView = mergedRnnOutView[:, :mergedVecLen//4]
						mergedRnnOutBackView = mergedRnnOutView[:, mergedVecLen//4:mergedVecLen//2]
				else:
					if SharedModel.args.attn == 0:
						mergedRnnOutFwdView = torch.cat( [mergedRnnOutView[:, :mergedVecLen//4], mergedRnnOutView[:, mergedVecLen//2:mergedVecLen*3//4]], 1 )
						mergedRnnOutBackView = torch.cat( [mergedRnnOutView[:, mergedVecLen//4:mergedVecLen//2], mergedRnnOutView[:, mergedVecLen*3//4:]], 1 )
					elif SharedModel.args.attn == 1:
						mergedRnnOutFwdView = torch.cat( [mergedRnnOutView[:, :mergedVecLen//8], mergedRnnOutView[:, mergedVecLen//4:mergedVecLen*3//8]], 1 )
						mergedRnnOutBackView = torch.cat( [mergedRnnOutView[:, mergedVecLen//8:mergedVecLen//4], mergedRnnOutView[:, mergedVecLen*3//8:mergedVecLen*1//2]], 1 )
					else:
						mergedRnnOutFwdView = torch.cat( [mergedRnnOutView[:, :mergedVecLen//8], mergedRnnOutView[:, mergedVecLen//2:mergedVecLen*5//8]], 1 )
						mergedRnnOutBackView = torch.cat( [mergedRnnOutView[:, mergedVecLen//8:mergedVecLen//4], mergedRnnOutView[:, mergedVecLen*5//8:mergedVecLen*3//4]], 1 )

				mergedRnnOutFwdView = self.reconFwdPrjLayer(mergedRnnOutFwdView)
				mergedRnnOutBackView = self.reconFwdPrjLayer(mergedRnnOutBackView)

				if SharedModel.args.recon_use_utt == True:
					mergedUttEncPrj = self.uttEncPrjLayer(mergedUttEncOut)
					mergedUttEncPrj = mergedUttEncPrj.expand(mergedRnnOut.size(0), mergedRnnOut.size(1), SharedModel.args.hid_dim).contiguous().view(-1, SharedModel.args.hid_dim)	# how about sum this instead of concat?
					mergedRnnOutFwdView = torch.cat([mergedUttEncPrj, mergedRnnOutFwdView], 1)
					mergedRnnOutBackView = torch.cat([mergedUttEncPrj, mergedRnnOutBackView], 1)

				finalOut.append(self.reconFwdOutLayer(mergedRnnOutFwdView))
				finalOut.append(self.reconBackOutLayer(mergedRnnOutBackView))
			#~

		if SharedModel.args.tag == True:
			if self.training == False or (SharedModel.domainCnt >= 2 and SharedModel.domainIdx == 0 and (SharedModel.args.tag_first_train_mb_cnt == -1 or curDmMbIdx < SharedModel.args.tag_first_train_mb_cnt)) or ((SharedModel.domainCnt >= 3 and SharedModel.domainIdx >= 1 and SharedModel.domainIdx < SharedModel.domainCnt-1) and (SharedModel.args.tag_mid_train_mb_cnt == -1 or curDmMbIdx < SharedModel.args.tag_mid_train_mb_cnt)) or (SharedModel.domainIdx == SharedModel.domainCnt-1 and (SharedModel.args.tag_last_train_mb_cnt == -1 or curDmMbIdx < SharedModel.args.tag_last_train_mb_cnt)):
				finalOut.append( self.tagOutLayer(mergedRnnOut.view(mergedRnnOut.size(0) * mergedRnnOut.size(1), mergedRnnOut.size(2))) )	# tagging output

		return finalOut
