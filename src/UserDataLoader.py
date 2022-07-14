import os
import shutil
import cv2
import sys
import argparse
from WordSegmentation import wordSegmentation, prepareImg

class FilePaths:
    """ Filenames and paths to data. """
    fnCharList = '../model/charList.txt' # символы
    fnAccuracy = '../model/accuracy.txt' # запись акураси
    fnTrain = '../data/train/' # место хранения данных
    fnWords = '../data/words/' # место и img распознаного текста (тест)
    fnWordsFromLines = '../data/words/fromlines_words/' #  место и img распознаного текста (тест)
    fnLines = '../data/lines/' # место для строк
    fnTexts = '../data/texts/' # место для текста
    fnCorpus = '../data/corpus.txt' # список слов
    fnDumpRes = '../dump/results.txt'  # где хранится нейронка

def segment_to_words(file_path):
    """сегментация из сторк в слова.
        Returns:
                все найденные слова в директории
    """
    # получить имя файла (imgFiles)
    if os.path.isdir(file_path):
        imgFiles = os.listdir(file_path)
    else:
        imgFiles = file_path

    found_words = []
    for (i, f) in enumerate(imgFiles):
        print("File #", i, " Name: ", f)
        print('Segmenting words of sample %s' % f)
        # проверить требования к в файлу
        if not check_file("%s/%s" % (file_path,f)):
            continue
        img = prepareImg(cv2.imread('%s%s' % (file_path, f)), 50)
        # execute segmentation with given parameters
        # -kernelSize: size of filter kernel (odd integer)
        # -sigma: standard deviation of Gaussian function used for filter kernel
        # -theta: approximated width/height ratio of words, filter function is distorted by this factor
        # - minArea: ignore word candidates smaller than specified area
        # Returns: List of tuples. Each tuple contains the bounding box and the image of the segmented word.
        tmp_words = wordSegmentation(img, kernelSize=25, sigma=11, theta=7, minArea=200)
        found_words.append(tmp_words)
    return found_words


def save_tmp_data(data, path, num, dtype):
    """ Сохранить найденные строки/слов в текст
        data - список/массив для сохранения
        path - дир для сохранения
        num - номер файла откуда производилась сгментация
        dtype - тип
    """
    if not os.path.exists('%s' % path):
        os.mkdir('%s' % path)

    if dtype not in ['word', 'line']:
        raise ValueError("dtype should be in 'word' or 'line'")

    if dtype == 'word':
        # Iterate over all segmented words
        print('Segmented into %d words' % len(data))
        for (j, w) in enumerate(data):
            for (k, n) in enumerate(w):
                (wordBox, wordImg) = n
                (x, y, w, h) = wordBox  # To draw bounding box in summary image (if needed)
                fn = '%s/t%d_w%d_%d.png' % (path, num, j, k)
                cv2.imwrite(fn, wordImg)  # save word

    elif dtype == 'line':
        # iterate over all segmented lines
        print('Segmented into %d lines' % len(data))
        for (j, w) in enumerate(data):
            fn = '%s/%d.png' % (path, j)
            cv2.imwrite(fn, w)  # save line


def clear_dirs(dtype):
    if dtype not in ['texts', 'lines']:
        raise ValueError("dtype should be in 'texts' or 'lines'")

    if dtype == 'texts':
        dirs = os.listdir(FilePaths.fnLines)
        for d in dirs:
            shutil.rmtree("%s%s" % (FilePaths.fnLines, d))
        dirs = os.listdir(FilePaths.fnWords)
        for d in dirs:
            shutil.rmtree("%s%s" % (FilePaths.fnWords, d))

    elif dtype == 'lines':
        files = os.listdir(FilePaths.fnLines)
        for f in files:
            if not f.endswith(".png"):
                shutil.rmtree('%s%s' % (FilePaths.fnLines, f))

        shutil.rmtree(FilePaths.fnWords)
        os.mkdir('%s' % FilePaths.fnWords)
    print("Directions & Files Removed")

def check_file(file_path):
    if not os.path.getsize(file_path):
        print("Warning, damaged images found: %s" % file_path)
        return False
    if not file_path.endswith(".png") or file_path.endswith(".PNG"):
        print("Warning, unsupported or invalid format of image found: %s" % file_path)
        return False
    # Image is OK
    return True
