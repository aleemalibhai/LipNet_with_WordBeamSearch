import os
import glob
import numpy as np
import cv2
from multiprocessing import Pool
import pdb
from torch.utils.data import DataLoader, Dataset
import time
from ffmpy import FFmpeg


class MyDataset(Dataset):
    
    def __init__(self):

        self.path = "/scratch/aalibhai/pretrain"
        self.out_path = "/scratch/aalibhai/LRS3"
        self.files = []

        folders = list(os.listdir(self.path))
        for folder in folders:
            files = list(os.listdir(os.path.join(self.path, folder)))
            for file in files:
                if (os.path.splitext(file)[1] == ".mp4"):
                    self.files.append(os.path.join(self.path, folder, file))


    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        file = self.files[idx]
        _, ext = os.path.splitext(file)
        dst = file.replace(self.path, self.out_path).replace(ext, '')

        if(not os.path.exists(dst)): 
            os.makedirs(dst)

        cmd = 'ffmpeg -i \'{}\' -qscale:v 2 -r 25 \'{}/%d.jpg\''.format(file, dst)
        os.system(cmd)
        
        return dst

if(__name__ == '__main__'):   
    dataset = MyDataset()
    loader = DataLoader(dataset, num_workers=32, batch_size=128, shuffle=False, drop_last=False)
    tic = time.time()
    for (i, batch) in enumerate(loader):
        eta = (1.0*time.time()-tic)/(i+1) * (len(loader)-i)
        print('eta:{}'.format(eta/3600.0))
