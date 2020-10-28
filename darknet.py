#Importing the libraries
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def parse_cfg(cfgfile):
	"""
	Takes a configuration file

	Returns a list of blocks. Each blocks describes a block in the neural
	network to be built. Block is represented as a dictionary in the list

	"""
	file = open(cfgfile,'r')
	lines = file.read().split('\n')
	lines = [x for x in lines if len(x) > 0]
	lines = [x for x in lines if x[0] != '#']
	lines = [x.rstrip().lstrip() for x in lines]

	blocks = []
	block = {}

	for line in lines:
		# the [ denotes the start of a new block.
		if line[0] == '[':
			if len(block) != 0:
				blocks.append(block)
				block = {}
			block['type'] = line[1:-1].rstrip()
		else:
			key, value = line.split('=')
			block[key.rstrip()] = value.lstrip()
	blocks.append(block)

	return blocks


