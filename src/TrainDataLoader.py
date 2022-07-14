from __future__ import division
from __future__ import print_function

import os
import random
import numpy as np
import cv2
from LinesSegmentation import normalize
from DataUtils import UtilsFilePaths
from UserDataLoader import FilePaths

class Sample:

    def __init__(self, gtText, filePath):
        self.gtText = gtText
        self.filePath = filePath


class Batch:

    def __init__(self, gtTexts, imgs):
        # Stack images over first axis
        self.imgs = np.stack(imgs, axis=0)
        self.gtTexts = gtTexts


class DataLoader:

    def __init__(self, filePath, batchSize, imgSize, maxTextLen):
        assert filePath[-1] == '/'

        self.dataAugmentation = False
        self.currIdx = 0
        self.batchSize = batchSize
        self.imgSize = imgSize
        self.samples = []
        f = open(filePath + 'words.txt')
        chars = set()

        bad_samples = []
        bad_samples_reference = []


        for line in f:
            if not line or line[0] == '#':
                continue
            lineSplit = line.strip().split(' ')

            assert len(lineSplit) >= 2

            fileNameSplit = lineSplit[0].split('-')
            fileName = filePath + fileNameSplit[0] + '/' + fileNameSplit[0] + '-' + fileNameSplit[1] + '.png'

            gtText = self.truncateLabel(' '.join(lineSplit[1:]), maxTextLen)
            chars = chars.union(set(list(gtText)))


            if not os.path.getsize(fileName):
                bad_samples.append(lineSplit[0] + '.png')
                continue

            self.samples.append(Sample(gtText, fileName))

        if set(bad_samples) != set(bad_samples_reference):
            print("Warning, damaged images found:", bad_samples)
            print("Damaged images expected:", bad_samples_reference)

        splitIdx = int(0.95 * len(self.samples))
        self.trainSamples = self.samples[:splitIdx]
        self.validationSamples = self.samples[splitIdx:]

        self.trainWords = [x.gtText for x in self.trainSamples]
        self.validationWords = [x.gtText for x in self.validationSamples]

        self.numTrainSamplesPerEpoch = 20000

        self.trainSet()

        self.charList = sorted(list(chars))


    def truncateLabel(self, text, maxTextLen):
        cost = 0
        for i in range(len(text)):
            if i != 0 and text[i] == text[i - 1]:
                cost += 2
            else:
                cost += 1
            if cost > maxTextLen:
                return text[:i]
        return text


    def trainSet(self):

        self.dataAugmentation = True
        self.currIdx = 0
        random.shuffle(self.trainSamples)
        self.samples = self.trainSamples[:self.numTrainSamplesPerEpoch]


    def validationSet(self):
        self.dataAugmentation = False
        self.currIdx = 0
        self.samples = self.validationSamples


    def getIteratorInfo(self):
        return (self.currIdx // self.batchSize + 1, len(self.samples) // self.batchSize)


    def hasNext(self):
        return self.currIdx + self.batchSize <= len(self.samples)


    def getNext(self):
        batchRange = range(self.currIdx, self.currIdx + self.batchSize)
        gtTexts = [self.samples[i].gtText for i in batchRange]
        imgs = [
            preprocess(cv2.imread(self.samples[i].filePath, cv2.IMREAD_GRAYSCALE), self.imgSize, self.dataAugmentation)
            for i in batchRange]
        self.currIdx += self.batchSize
        return Batch(gtTexts, imgs)


def preprocess(img, imgSize, dataAugmentation=False):
    if img is None:
        img = np.zeros([imgSize[1], imgSize[0]])

    if dataAugmentation:
        stretch = (random.random() - 0.5)
        wStretched = max(int(img.shape[1] * (1 + stretch)), 1)
        img = cv2.resize(img, (wStretched, img.shape[0]))

    (wt, ht) = imgSize
    (h, w) = img.shape
    fx = w / wt
    fy = h / ht
    f = max(fx, fy)
    newSize = (max(min(wt, int(w / f)), 1),
               max(min(ht, int(h / f)), 1))
    img = cv2.resize(img, newSize)
    target = np.ones([ht, wt]) * 255
    target[0:newSize[1], 0:newSize[0]] = img

    img = cv2.transpose(target)

    # normalize
    return normalize(img)

