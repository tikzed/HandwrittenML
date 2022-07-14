from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import tensorflow as tf
import os

class ModelFilePaths:
	modelDir = '../model/model'
	dumpDir = '../dump/'
	snapshotsDir = '../model/snapshot'


class DecoderType:
	BestPath = 0
	BeamSearch = 1


class Model: 

	batchSize = 50
	imgSize = (128, 32)
	maxTextLen = 32

	def __init__(self, charList, decoderType=DecoderType.BestPath, mustRestore=False, dump=False):
		self.dump = dump
		self.charList = charList
		self.decoderType = decoderType
		self.mustRestore = mustRestore
		self.snapID = 0
		self.is_train = tf.placeholder(tf.bool, name='is_train')
		self.inputImgs = tf.placeholder(tf.float32, shape=(None, Model.imgSize[0], Model.imgSize[1]))
		self.setupCNN()
		self.setupRNN()
		self.setupCTC()
		self.batchesTrained = 0
		self.learningRate = tf.placeholder(tf.float32, shape=[])
		self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

		with tf.control_dependencies(self.update_ops):
			self.optimizer = tf.train.RMSPropOptimizer(self.learningRate).minimize(self.loss)

		(self.sess, self.saver) = self.setupTF()

			
	def setupCNN(self):

		cnnIn4d = tf.expand_dims(input=self.inputImgs, axis=3)

		kernelVals = [5, 5, 3, 3, 3]
		featureVals = [1, 32, 64, 128, 128, 256]
		strideVals = poolVals = [(2,2), (2,2), (1,2), (1,2), (1,2)]
		numLayers = len(strideVals)

		pool = cnnIn4d
		for i in range(numLayers):

			kernel = tf.Variable(tf.truncated_normal([kernelVals[i], kernelVals[i], featureVals[i], featureVals[i + 1]], stddev=0.1))
			conv = tf.nn.conv2d(pool, kernel, padding='SAME',  strides=(1,1,1,1))
			conv_norm = tf.layers.batch_normalization(conv, training=self.is_train)
			relu = tf.nn.relu(conv_norm)
			pool = tf.nn.max_pool(relu, (1, poolVals[i][0], poolVals[i][1], 1), (1, strideVals[i][0], strideVals[i][1], 1), 'VALID')
		self.cnnOut4d = pool


	def setupRNN(self):
		rnnIn3d = tf.squeeze(self.cnnOut4d, axis=[2])

		numHidden = 256
		cells = [tf.contrib.rnn.LSTMCell(num_units=numHidden, state_is_tuple=True) for _ in range(2)] # 2 layers

		stacked = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

		((fw, bw), _) = tf.nn.bidirectional_dynamic_rnn(cell_fw=stacked, cell_bw=stacked, inputs=rnnIn3d, dtype=rnnIn3d.dtype)

		concat = tf.expand_dims(tf.concat([fw, bw], 2), 2)

		kernel = tf.Variable(tf.truncated_normal([1, 1, numHidden * 2, len(self.charList) + 1], stddev=0.1))

		self.rnnOut3d = tf.squeeze(tf.nn.atrous_conv2d(value=concat, filters=kernel, rate=1, padding='SAME'), axis=[2])
		

	def setupCTC(self):

		self.ctcIn3dTBC = tf.transpose(self.rnnOut3d, [1, 0, 2])

		self.gtTexts = tf.SparseTensor(tf.placeholder(tf.int64, shape=[None, 2]) , tf.placeholder(tf.int32, [None]), tf.placeholder(tf.int64, [2]))

		self.seqLen = tf.placeholder(tf.int32, [None])
		self.loss = tf.reduce_mean(tf.nn.ctc_loss(labels=self.gtTexts, inputs=self.ctcIn3dTBC, sequence_length=self.seqLen, ctc_merge_repeated=True))

		self.savedCtcInput = tf.placeholder(tf.float32, shape=[Model.maxTextLen, None, len(self.charList) + 1])
		self.lossPerElement = tf.nn.ctc_loss(labels=self.gtTexts, inputs=self.savedCtcInput, sequence_length=self.seqLen, ctc_merge_repeated=True)

		if self.decoderType == DecoderType.BestPath:
			self.decoder = tf.nn.ctc_greedy_decoder(inputs=self.ctcIn3dTBC, sequence_length=self.seqLen)
		elif self.decoderType == DecoderType.BeamSearch:
			self.decoder = tf.nn.ctc_beam_search_decoder(inputs=self.ctcIn3dTBC, sequence_length=self.seqLen, beam_width=50, merge_repeated=False)


	def setupTF(self):

		print('Python: ' + sys.version)
		print('Tensorflow: ' + tf.__version__)

		sess = tf.Session()
		saver = tf.train.Saver(max_to_keep=1)

		latestSnapshot = tf.train.latest_checkpoint(ModelFilePaths.modelDir)

		if self.mustRestore and not latestSnapshot:
			raise Exception('No saved model found in %s: ' % ModelFilePaths.modelDir)

		if latestSnapshot:
			print('Init with stored values from %s' % latestSnapshot)
			saver.restore(sess, latestSnapshot)
		else:
			print('Init with new values')
			sess.run(tf.global_variables_initializer())

		return (sess, saver)


	def toSparse(self, texts):
		indices = []
		values = []
		shape = [len(texts), 0]

		for (batchElement, text) in enumerate(texts):
			labelStr = [self.charList.index(c) for c in text]
			if len(labelStr) > shape[1]:
				shape[1] = len(labelStr)
			for (i, label) in enumerate(labelStr):
				indices.append([batchElement, i])
				values.append(label)

		return (indices, values, shape)


	def decoderOutputToText(self, ctcOutput, batchSize):
		encodedLabelStrs = [[] for i in range(batchSize)]

		decoded = ctcOutput[0][0]
		idxDict = { b : [] for b in range(batchSize) }
		for (idx, idx2d) in enumerate(decoded.indices):
			label = decoded.values[idx]
			batchElement = idx2d[0]
			encodedLabelStrs[batchElement].append(label)

		return [str().join([self.charList[c] for c in labelStr]) for labelStr in encodedLabelStrs]


	def trainBatch(self, batch):
		numBatchElements = len(batch.imgs)
		sparse = self.toSparse(batch.gtTexts)
		rate = 0.01 if self.batchesTrained < 10 else (0.001 if self.batchesTrained < 10000 else 0.0001)
		evalList = [self.optimizer, self.loss]
		feedDict = {self.inputImgs : batch.imgs, self.gtTexts : sparse , self.seqLen : [Model.maxTextLen] * numBatchElements, self.learningRate : rate, self.is_train: True}
		(_, lossVal) = self.sess.run(evalList, feedDict)
		self.batchesTrained += 1
		return lossVal


	def dumpNNOutput(self, rnnOutput):
		if not os.path.isdir(ModelFilePaths.dumpDir):
			os.mkdir(ModelFilePaths.dumpDir)

		maxT, maxB, maxC = rnnOutput.shape
		for b in range(maxB):
			csv = ''
			for t in range(maxT):
				for c in range(maxC):
					csv += str(rnnOutput[t, b, c]) + ';'
				csv += '\n'
			fn = ModelFilePaths.dumpDir + 'rnnOutput_' + str(b)+'.csv'
			print('Write dump of NN to file: ' + fn)

			with open(fn, 'w') as f:
				f.write(csv)


	def inferBatch(self, batch, calcProbability=False, probabilityOfGT=False):
		numBatchElements = len(batch.imgs)
		evalRnnOutput = self.dump or calcProbability
		evalList = [self.decoder] + ([self.ctcIn3dTBC] if evalRnnOutput else [])
		feedDict = {self.inputImgs : batch.imgs, self.seqLen : [Model.maxTextLen] * numBatchElements, self.is_train: False}
		evalRes = self.sess.run(evalList, feedDict)
		decoded = evalRes[0]
		texts = self.decoderOutputToText(decoded, numBatchElements)

		probs = None
		if calcProbability:
			sparse = self.toSparse(batch.gtTexts) if probabilityOfGT else self.toSparse(texts)
			ctcInput = evalRes[1]
			evalList = self.lossPerElement
			feedDict = {self.savedCtcInput : ctcInput, self.gtTexts : sparse, self.seqLen : [Model.maxTextLen] * numBatchElements, self.is_train: False}
			lossVals = self.sess.run(evalList, feedDict)
			probs = np.exp(-lossVals)

		if self.dump:
			self.dumpNNOutput(evalRes[1])

		return (texts, probs)
	

	def save(self):
		self.snapID += 1
		self.saver.save(self.sess, ModelFilePaths.snapshotsDir, global_step=self.snapID)
 
