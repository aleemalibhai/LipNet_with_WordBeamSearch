
import os
import inflect

p = inflect.engine()
count =0
nerr=[]
path = '/scratch/aalibhai/pretrain/'
with open('/home/aalibhai/Thesis/LipNet-CTCWBS/data/LRS3_train.txt', 'r') as f:
    lines = [line.strip() for line in f.readlines()]

    for line in lines:
        with open(os.path.join(path, line+'.txt'), 'r')as k:
            lines = [line.strip().split(' ') for line in k.readlines()]
            lines = lines[4:]
            txt = [line[0].lstrip() for line in lines if float(line[2]) < 3.0]
            txt = list(filter(lambda s: not s.upper() in ['{NS}', '{LG}'], txt))
            for i, _ in enumerate(txt):
                if not txt[i].isalpha():
                    txt[i] = p.number_to_words(txt[i]).upper().replace('-', ' ').replace(',', '')
            x = sum(len(i) for i in txt)
            if x == 0:
                print(line)

# with open('/home/aalibhai/Thesis/LipNet-CTCWBS/data/LRS3_train.txt', 'w') as f:
#     for line in nerr:
#         f.write(line + '\n')


