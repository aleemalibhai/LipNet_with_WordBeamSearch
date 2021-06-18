import sys
import os
import cv2
import face_alignment
from multiprocessing import Pool, Process, Queue
import time


def run(gpu, files):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cuda')
    print('gpu={},n_files={}'.format(gpu, len(files)))
    tic = time.time()
    count = 0
    for (img_name, savename) in files:
        I = cv2.imread(img_name)
        points_list = fa.get_landmarks(I)
        
        with open(savename, 'w') as f:
            if(points_list is not None):
                for points in points_list:
                    for (x, y) in points:
                        f.write('({}, {})\t'.format(x, y))
                    f.write('\n')

        count += 1
        if(count % 1000 == 0):
            print('dst={},eta={}'.format(savename, (time.time()-tic)/(count) * (len(files) - count) / 3600.0))
       

if(__name__ == '__main__'):
    in_path = '/scratch/aalibhai/LRS3'
    out_path = '/scratch/aalibhai/LRS3_annos'
    data = []
    count = 0
    print('making file list')
    for speaker in os.listdir(in_path):
        for vid in os.listdir(os.path.join(in_path, speaker)):
            for img in os.listdir(os.path.join(in_path, speaker, vid)):
                data.append((
                    os.path.join(in_path, speaker, vid, img),
                    os.path.join(out_path, speaker, vid, img.replace('.jpg', '.txt'))
                    ))

    print('file list created')
    
    
    for (_, dst) in data:
        dir, _ = os.path.split(dst)
        if(not os.path.exists(dir)):
            os.makedirs(dir)
       
    processes = []
    n_p = 3
    gpus = ['1', '2', '3']
    bs = len(data) // n_p
    for i in range(n_p):
        if(i == n_p - 1):
            bs = len(data)
        print('starting proc')
        p = Process(target=run, args=(gpus[i],data[:bs],))
        print('proc startd')
        data = data[bs:]
        p.start()
        processes.append(p)
    assert(len(data) == 0)
    for p in processes:
        p.join()
