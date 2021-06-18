# encoding: utf-8
import numpy as np
import glob
import time
import cv2
import os
from torch.utils.data import Dataset
from cvtransforms import *
import torch
import glob
import re
import copy
import json
import random
import editdistance
import inflect

    
class MyDataset(Dataset):
    letters = [' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    def __init__(self, video_path, anno_path, file_list, vid_pad, txt_pad, phase):
        self.anno_path = anno_path
        self.vid_pad = vid_pad
        self.txt_pad = txt_pad
        self.phase = phase
        
        with open(file_list, 'r') as f:
            self.videos = [os.path.join(video_path, line.strip()) for line in f.readlines()]
            
        self.data = []
        for vid in self.videos:
            if video_path.find('LRS3_lip') == -1:
                items = vid.split(os.path.sep)            
                self.data.append((vid, items[-4], items[-1]))
            else:
                if (len(os.listdir(vid)) > 0):
                    items = vid.split(os.path.sep)
                    self.data.append((vid, items[-2], items[-1]))        
                
    def __getitem__(self, idx):
        (vid, spk, name) = self.data[idx]
        vid = self._load_vid(vid)
        if self.anno_path.find('GRID') != -1:
            anno = self._load_anno(os.path.join(self.anno_path, spk, 'align', name + '.align'))
        else:
            anno = self._load_anno(os.path.join(self.anno_path, spk, name + '.txt'))

        if(self.phase == 'train'):
            vid = HorizontalFlip(vid)
          
        vid = ColorNormalize(vid)                   
        if len(anno) < 1:
            print(os.path.join(spk, name))
            print(anno)
        vid_len = vid.shape[0]
        anno_len = anno.shape[0]
        vid = self._padding(vid, self.vid_pad)
        anno = self._padding(anno, self.txt_pad, spk + '/' + name)

        
        return {'vid': torch.FloatTensor(vid.transpose(3, 0, 1, 2)), 
            'txt': torch.LongTensor(anno),
            'txt_len': anno_len,
            'vid_len': vid_len}
            
    def __len__(self):
        return len(self.data)
        
    def _load_vid(self, p): 
        files = os.listdir(p)
        files = list(filter(lambda file: file.find('.jpg') != -1, files))
        files = sorted(files, key=lambda file: int(os.path.splitext(file)[0]))
        if len(files) > 75:
            files = files[:75]
        array = [cv2.imread(os.path.join(p, file)) for file in files]
        array = list(filter(lambda im: not im is None, array))
        array = list(filter(lambda im: len(im[im == np.nan]) == 0, array))
        array = [cv2.resize(im, (128, 64), interpolation=cv2.INTER_LANCZOS4) for im in array]
        try:
            array = np.stack(array, axis=0).astype(np.float32)
        except:
            print(p)
        return array
    
    def _load_anno(self, name):
        p = inflect.engine()
        with open(name, 'r') as f:
            if self.anno_path.find('GRID') != -1:
                lines = [line.strip().split(' ') for line in f.readlines()]
                txt = [line[2] for line in lines]
                txt = list(filter(lambda s: not s.upper() in ['SIL', 'SP'], txt))
            else:
                lines = [line.strip().split(' ') for line in f.readlines()]
                lines = lines[4:]
                txt = [line[0].lstrip() for line in lines if float(line[2]) < 3.0]
                txt = list(filter(lambda s: not s.upper() in ['{NS}', '{LG}'], txt))
                for i, _ in enumerate(txt):
                    if not txt[i].isalpha():
                        txt[i] = p.number_to_words(txt[i]).upper().replace('-', ' ').replace(',', '').replace("'", '')

        return MyDataset.txt2arr(' '.join(txt).upper(), 1)
    
    def _padding(self, array, length, name='boop'):

        array = [array[_] for _ in range(array.shape[0])]
        size = array[0].shape
        for i in range(length - len(array)):
            array.append(np.zeros(size))
        return np.stack(array, axis=0)
    
    @staticmethod
    def txt2arr(txt, start):
        arr = []
        for c in list(txt):
            arr.append(MyDataset.letters.index(c) + start)
        return np.array(arr)
        
    @staticmethod
    def arr2txt(arr, start):
        txt = []
        for n in arr:
            if(n >= start):
                txt.append(MyDataset.letters[n - start])     
        return ''.join(txt).strip()
    
    @staticmethod
    def ctc_arr2txt(arr, start):
        pre = -1
        txt = []
        for n in arr:
            if(pre != n and n >= start):                
                if(len(txt) > 0 and txt[-1] == ' ' and MyDataset.letters[n - start] == ' '):
                    pass
                else:
                    txt.append(MyDataset.letters[n - start])                
            pre = n
        return ''.join(txt).strip()
            
    @staticmethod
    def wer(predict, truth):      
        word_pairs = [(p[0].split(' '), p[1].split(' ')) for p in zip(predict, truth)]
        wer = [1.0*editdistance.eval(p[0], p[1])/len(p[1]) for p in word_pairs]
        return wer
        
    @staticmethod
    def cer(predict, truth):        
        cer = [1.0*editdistance.eval(p[0], p[1])/len(p[1]) for p in zip(predict, truth)]
        return cer
