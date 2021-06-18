import os
import random
import inflect

path = '/scratch/aalibhai/LRS3_lip'
apath = '/scratch/aalibhai/pretrain'
corpus = '/home/aalibhai/Thesis/LipNet-CTCWBS/LRS3_corpus.txt'
train_list = '/home/aalibhai/Thesis/LipNet-CTCWBS/data/LRS3_train.txt'
val_list = '/home/aalibhai/Thesis/LipNet-CTCWBS/data/LRS3_val.txt'
vids = []
txt = []
vids_val = []
p = inflect.engine()
count = 0
print('making file list')
for speaker in os.listdir(path):
    for vid in os.listdir(os.path.join(path, speaker)):
        if len(os.listdir(os.path.join(path, speaker, vid))) > 0:
            lines= []
            with open(os.path.join(apath, speaker, vid + '.txt')) as f:
                lines = [line.strip().split(' ') for line in f.readlines()]
                lines = lines[4:]
                txt = [line[0].lstrip() for line in lines if float(line[2]) < 3.0]
                txt = list(filter(lambda s: not s.upper() in ['{NS}', '{LG}'], txt))
                for i, _ in enumerate(txt):
                    if not txt[i].isalpha():
                        txt[i] = p.number_to_words(txt[i]).upper().replace('-', ' ').replace(',', '')

                if len(txt) > 0 and sum(len(i) for i in txt) <= 75 and sum(len(i) for i in txt) > 0:
                    vids.append(os.path.join(speaker, vid))
        count += 1
        if count % 1000 == 0:
            print(count, end='\r')

count = 0

print('splitting train/validation')
num = range(len(vids)//10)
for i in num:
    vids_val.append(vids.pop(random.randrange(len(vids))))

print('writing train list')
count = 0
l = len(vids)
with open(train_list, 'w') as t:
    for file in vids:
        t.write(file + '\n')
        count += 1
        if count % 100000 == 0:
            print('{}/{}'.format(count, l), end='\r')

print('writing validation list')
count = 0
l = len(vids_val)
with open(val_list, 'w') as v:
    for file in vids_val:
        v.write(file + '\n')
        count += 1
        if count % 100000 == 0:
            print('{}/{}'.format(count, l), end='\r')