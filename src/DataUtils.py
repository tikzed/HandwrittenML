from __future__ import division
from __future__ import print_function

import os
import re
import io
import random
import numpy as np
import cv2


class UtilsFilePaths:
    fnTrain = '../data/train/'
    fnInfo = '../data/train/words.txt'


def main():
    write_infofile = True
    rename = False
    subdirs = os.listdir(UtilsFilePaths.fnTrain)

    if rename:
        for num, sb in enumerate(subdirs):
            renameFiles("%s%s" % (UtilsFilePaths.fnTrain, sb))

    if write_infofile:
        createDiscriptionFile(UtilsFilePaths.fnTrain, UtilsFilePaths.fnInfo)


def createDiscriptionFile (dirpath, fpath) :

    strs_to_write = []
    subdirs = os.listdir(dirpath)
    words_txt = _searchTxt(subdirs)

    if words_txt != 'None':
        subdirs.remove(words_txt)
    else :
        print("No files with .txt extension found in direction: %s" % dirpath)

    for num, sub in enumerate(subdirs):
        subdir_str = "%s-" % sub

        files = os.listdir("%s%s" % (dirpath, sub))
        file = _searchTxt(files)
        if not file:
            print("No files with .txt extension found in direction: %s" % sub)

        file_path = "%s%s/%s" % (dirpath, sub, file)

        file_handler =  io.open(file_path, 'r', encoding='utf-8')
        str_to_write = ""
        while True:
            line = file_handler.readline()
            if not line:
                break
            str_to_write += "%s%s" % (subdir_str, line)
        strs_to_write.append(str_to_write)
        file_handler.close()

    print("Ended")


def _searchTxt(files):
    for num, f in enumerate(files):
        if f.endswith(".txt"):
            return f
    return 'None'

def renameFiles(dir):
    count = 0
    files = os.listdir(dir)
    for num, f in enumerate(files):
        match = re.search(r'\d\d\d\D\d*\.png', f)
        if  match:
            continue
        elif f.endswith(".txt"):
            continue
        else:
            os.rename("%s/%s" % (dir, f), "%s/%s-%s" % (dir, dir.split('/')[3], f))
            print("Renamed from %s/%s to %s/%s-%s" % (dir, f, dir, dir.split('/')[3], f))
            count = count+1
    print("Total, renamed: %d files in dir:%s" % (count, dir))

if __name__ == '__main__':
    main()