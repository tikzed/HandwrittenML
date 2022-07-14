import os
import shutil
import cv2
import sys
import argparse
import editdistance
from LinesSegmentation import lineSegmentation
from WordSegmentation import wordSegmentation, prepareImg
from UserDataLoader import segment_to_words, save_tmp_data, clear_dirs, FilePaths, check_file
from TrainDataLoader import DataLoader, Batch, preprocess
from Model import Model, DecoderType, ModelFilePaths


def main():
    """
        парсинг аргументов
        создание нейронки
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', help='image - is a full text consist of many lines', action='store_true') #, default=True
    parser.add_argument('--line', help='image - is line with words', action='store_true') #, default=False
    parser.add_argument('--train', help='train the NN', action='store_true')
    parser.add_argument('--validate', help='validate the NN', action='store_true')
    parser.add_argument('--beamsearch', help='use beam search instead of best path decoding', action='store_true')
    parser.add_argument('--dump', help='dump output of NN to CSV file(s)', action='store_true')
    args = parser.parse_args()


    if args.text:
        clear_dirs("texts")
        # Порвка директории
        imgFiles = os.listdir(FilePaths.fnTexts)
        if not imgFiles:
            print("Error! No files found in data dir:%s" % FilePaths.fnTexts)
            return -1
        print("Files found in data dir:{0}".format(len(imgFiles)))
        for (i, f) in enumerate(imgFiles):
            print("File #", i, " Name: ", f)
            print('Segmenting lines of sample %s' % f)
            if not check_file("%s/%s" % (FilePaths.fnTexts, f)):
                continue
            img = cv2.imread('%s%s' % (FilePaths.fnTexts, f))
            tmp_lines = lineSegmentation(img)
            fpath = ("%s/text%d_lines/" % (FilePaths.fnLines, i))
            save_tmp_data(tmp_lines, fpath, i, dtype='line')

            res_words = segment_to_words(fpath)
            wfpath = ("%s/text%d_words/" % (FilePaths.fnWords, i))
            save_tmp_data(res_words, wfpath, i, dtype='word')

    elif args.line:
        clear_dirs("lines")

        imgFiles = os.listdir(FilePaths.fnLines)
        print("Files found in data dir:  {0}".format(len(imgFiles)))

        res_words = segment_to_words(FilePaths.fnLines)
        wfpath = ("%s/fromlines_words/" % FilePaths.fnWords)
        save_tmp_data(res_words, wfpath, 0, dtype='word')

    decoderType = DecoderType.BestPath
    if args.beamsearch:
        decoderType = DecoderType.BeamSearch

    if args.train or args.validate:
        loader = DataLoader(FilePaths.fnTrain, Model.batchSize, Model.imgSize, Model.maxTextLen)
        open(FilePaths.fnCharList, 'w').write(str().join(loader.charList))
        open(FilePaths.fnCorpus, 'w').write(str(' ').join(loader.trainWords + loader.validationWords))
        if args.train:
            model = Model(loader.charList, decoderType)
            train(model, loader)
        elif args.validate:
            model = Model(loader.charList, decoderType, mustRestore=True)
            validate(model, loader)
    else:
        if os.path.exists('%s' % FilePaths.fnAccuracy):
            print(open(FilePaths.fnAccuracy).read())
        model = Model(open(FilePaths.fnCharList).read(), decoderType, mustRestore=True, dump=args.dump)

        if args.text:
            dirs = os.listdir(FilePaths.fnWords)
            if not dirs:
                print("No directions with words found in data direction: %s" % FilePaths.fnWords)
                return -1
            for (i, d) in enumerate(dirs):
                print("Direction from text #", i, " Named: ", d)
                print('Executing files from %s...' % d)
                files = os.listdir("%s%s/" % (FilePaths.fnWords, d))
                if not files:
                    print("No words found in data direction: %s" % ("%s%s/" % (FilePaths.fnWords, d)))
                    continue
                infer(model, "%s%s/" % (FilePaths.fnWords, d))

        elif args.line:
            files = os.listdir(FilePaths.fnWordsFromLines)
            if not files:
                print("No files with words found in data direction: %s" % FilePaths.fnWordsFromLines)
                return -1
            infer(model, FilePaths.fnWordsFromLines)


def train(model, loader):
    epoch = 0
    bestCharErrorRate = float('inf')
    noImprovementSince = 0
    earlyStopping = 3
    while True:
        epoch += 1
        print('Epoch:', epoch)

        print('Train NN')
        loader.trainSet()

        while loader.hasNext():
            iterInfo = loader.getIteratorInfo()
            batch = loader.getNext()
            loss = model.trainBatch(batch)
            print('Batch:', iterInfo[0], '/', iterInfo[1], 'Loss:', loss)

        charErrorRate = validate(model, loader)

        if charErrorRate < bestCharErrorRate:
            print('Character error rate improved: %f%%, save model...' % (charErrorRate * 100.0))
            bestCharErrorRate = charErrorRate
            noImprovementSince = 0
            model.save()
            open(FilePaths.fnAccuracy, 'w').write(
                'Validation character error rate of saved model: %f%%' % (charErrorRate * 100.0))
        else:
            print('Character error rate not improved')
            noImprovementSince += 1

        if noImprovementSince >= earlyStopping:
            print('No more improvement since %d epochs. Training stopped.' % earlyStopping)
            break


def validate(model, loader):
    print('Validate NN')
    loader.validationSet()
    numCharErr = 0
    numCharTotal = 0
    numWordOK = 0
    numWordTotal = 0
    while loader.hasNext():
        iterInfo = loader.getIteratorInfo()
        print('Batch:', iterInfo[0], '/', iterInfo[1])
        batch = loader.getNext()
        (recognized, _) = model.inferBatch(batch)

        print('Ground truth -> Recognized')
        for i in range(len(recognized)):
            numWordOK += 1 if batch.gtTexts[i] == recognized[i] else 0
            numWordTotal += 1
            dist = editdistance.eval(recognized[i], batch.gtTexts[i])
            numCharErr += dist
            numCharTotal += len(batch.gtTexts[i])
            print('[OK]' if dist == 0 else '[ERR:%d]' % dist, '"' + batch.gtTexts[i] + '"', '->',
                  '"' + recognized[i] + '"')

    charErrorRate = numCharErr / numCharTotal
    wordAccuracy = numWordOK / numWordTotal
    print('Character error rate: %f%%. Word accuracy: %f%%.' % (charErrorRate * 100.0, wordAccuracy * 100.0))
    return charErrorRate

def infer(model, fpath):

    if os.path.isdir(fpath):
        imgFiles = os.listdir(fpath)
    else:
        imgFiles = fpath
    recognized_words = []
    for (i, fnImg) in enumerate(imgFiles):
        print("File #", i, " Name: ", fnImg)
        print('Recognizing text from image %s...' % fnImg)
        if not check_file("%s/%s" % (fpath, fnImg)):
            continue
        img = preprocess(cv2.imread('%s%s' % (fpath, fnImg), cv2.IMREAD_GRAYSCALE), Model.imgSize)
        batch = Batch(None, [img])
        (recognized, probability) = model.inferBatch(batch, False)
        recognized_words.append(recognized[0])
        print('Recognized:', '"' + recognized[0] + '"')
        if probability:
            print('Probability:', probability[0])

    dump_results(recognized_words)


def dump_results(res):

    if not os.path.isdir(ModelFilePaths.dumpDir):
        os.mkdir(ModelFilePaths.dumpDir)
    str_to_write = str(' ').join(res) + "\n"
    with open(FilePaths.fnDumpRes, 'a+') as f:
        f.write(str_to_write)



if __name__ == '__main__':
	main()