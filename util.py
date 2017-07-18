# -------------------------------------------------------
# Joo-Kyung Kim (kimjook@cse.ohio-state.edu)	3/28/2017
# -------------------------------------------------------

import torch
import torch.nn as nn


def loadIdx(idxFile, padAtZeroIdx=False):
	idxs = {}
	idxList = []

	i = 0
	if padAtZeroIdx == True:
		idxs['<pad>'] = 0
		idxList.append('<pad>')
		i = 1

	for line in open(idxFile):
		line = line.strip()
		idxs[line] = i
		idxList.append(line)
		i += 1

	return idxs, idxList


def unitNorm(input):
	denom = input.data.norm(2, 1).add_(1e-10).expand_as(input.data)
	input.data.div_( denom.expand_as(input) )


class NoisePassedDropoutFunc(torch.nn._functions.dropout.Dropout):	# noise is not generated here but passed as an argument
	def __init__(self, noise, p=0.5, train=False, inplace=False):
		super(NoisePassedDropoutFunc, self).__init__(p, train, inplace)
		self.noise = noise

	def forward(self, input):
		if self.inplace:
			self.mark_dirty(input)
			output = input
		else:
			output = input.clone()

		if self.p > 0 and self.train:
			output.mul_(self.noise)
		return output


class SeqConstDropout(nn.Dropout):
	def __init__(self, p=0.5, inplace=False, lazyMaskFlag=False):	# if lazyMaskFlag is True, noise should be assigned after init before forward
		super(SeqConstDropout, self).__init__(p, inplace)
		self.lazyMaskFlag = lazyMaskFlag

	def forward(self, input):
		if self.p > 0 and self.train:
			if self.lazyMaskFlag == False:
				self.noise = input.data.new().resize_(1, input.size(1), input.size(2))	# for timesteps X batches X dims inputs, let each time step has the same dropout mask
				self.noise.bernoulli_(1 - self.p).div_(1 - self.p)
				if self.p == 1:
					self.noise.fill_(0)
			self.seqNoise = self.noise.expand_as(input.data)

		return NoisePassedDropoutFunc(self.seqNoise, self.p, self.training, self.inplace)(input)



# Xavier initialization to the weights, forget bias 1, and other biases 0
def initLSTM(module, layerCnt, blstmFlag):
	modState = module.state_dict()

	suffixs = ['']
	if blstmFlag == True:
		suffixs.append('_reverse')

	for suffix in suffixs:
		for i in range(layerCnt):
			torch.nn.init.xavier_normal( modState['weight_ih_l'+str(i)+suffix] )
			torch.nn.init.xavier_normal( modState['weight_hh_l'+str(i)+suffix] )
			biasLen = int(modState['bias_ih_l'+str(i)+suffix].size(0) / 4)
			modState['bias_ih_l'+str(i)+suffix].zero_()
			modState['bias_hh_l'+str(i)+suffix].zero_()
			modState['bias_ih_l'+str(i)+suffix][biasLen : 2*biasLen].fill_(0.5)
			modState['bias_hh_l'+str(i)+suffix][biasLen : 2*biasLen].fill_(0.5)

	'''
	params = list(module.parameters())      # init the same as torch7's Element-RNN library
	#stdv = 1.0 / math.sqrt(SharedModel.args.emb_dim + SharedModel.args.hid_dim)

	#params[0].data.normal_(0, stdv)
	#params[1].data.normal_(0, stdv)
	torch.nn.init.xavier_normal(params[0].data)
	torch.nn.init.xavier_normal(params[1].data)
	params[2].data.zero_()
	params[3].data.zero_()
	params[2].data[hidDim : 2*hidDim].fill_(0.5)
	params[3].data[hidDim : 2*hidDim].fill_(0.5)

	if blstmFlag == True:
		#params[4].data.normal_(0, stdv)
		#params[5].data.normal_(0, stdv)
		torch.nn.init.xavier_normal(params[4].data)
		torch.nn.init.xavier_normal(params[5].data)
		params[6].data.zero_()
		params[7].data.zero_()
		params[6].data[hidDim : 2*hidDim].fill_(0.5)
		params[7].data[hidDim : 2*hidDim].fill_(0.5)
	'''
