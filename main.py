import torch
import pandas as pd
import codecs
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
import os
import sys
from dataset import MyDataset
import numpy as np
import time
from model import LipNet
import torch.optim as optim
import re
import json
from tensorboardX import SummaryWriter
from word_beam_search import WordBeamSearch


if(__name__ == '__main__'):
    opt = __import__('options')
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    writer = SummaryWriter()

def dataset2dataloader(dataset, shuffle=True):
    return DataLoader(dataset,
        batch_size = opt.batch_size,
        shuffle = shuffle,
        num_workers = opt.num_workers,
        drop_last = False)

def show_lr(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return np.array(lr).mean()

def ctc_decode(y):
    y = y.argmax(-1)
    return [MyDataset.ctc_arr2txt(y[_], start=1) for _ in range(y.size(0))]


def ctc_decode2(y, wbs):
    res1 = wbs.compute(y)
    chars = codecs.open('chars.txt', 'r', 'utf8').read()
    blank = len(chars)
    strings = []
    for res in res1:
        string = ''
        for label in res:
            if label == blank:
                break
            string += chars[label]  # map label to char
        strings.append(string.upper())
    return strings

def test(model, net):
    if (opt.is_wbs):
        wbs = init_word_beam_search()

    with torch.no_grad():
        dataset = MyDataset(opt.video_path,
            opt.anno_path,
            opt.train_list,
            opt.vid_padding,
            opt.txt_padding,
            'test')

        print('num_test_data:{}'.format(len(dataset.data)))
        model.eval()
        loader = dataset2dataloader(dataset, shuffle=False)
        loss_list = []
        wer = []
        cer = []
        crit = nn.CTCLoss(zero_infinity=True)
        tic = time.time()
        for (i_iter, input) in enumerate(loader):
            vid = input.get('vid').cuda()
            txt = input.get('txt').cuda()
            vid_len = input.get('vid_len').cuda()
            txt_len = input.get('txt_len').cuda()

            y = net(vid)
            loss = crit(y.transpose(0, 1).log_softmax(-1), txt, vid_len.view(-1), txt_len.view(-1)).detach().cpu().numpy()
            loss_list.append(loss)
            
            if (opt.is_wbs):
                pred_txt = ctc_decode2(y.permute(1, 0, 2).roll(-1, 2).softmax(-1).detach().cpu().numpy(), wbs)
            else:
                pred_txt = ctc_decode(y)
            
            truth_txt = [MyDataset.arr2txt(txt[_], start=1) for _ in range(txt.size(0))]
            wer.extend(MyDataset.wer(pred_txt, truth_txt))
            cer.extend(MyDataset.cer(pred_txt, truth_txt))
            if(i_iter % opt.display == 0):
                v = 1.0*(time.time()-tic)/(i_iter+1)
                eta = v * (len(loader)-i_iter) / 3600.0

                print(''.join(101*'-'))
                print('{:<75}|{:>75}'.format('predict', 'truth'))
                print(''.join(101*'-'))
                for (predict, truth) in list(zip(pred_txt, truth_txt))[:10]:
                    print('{:<75}|{:>75}'.format(predict, truth))
                print(''.join(101 *'-'))
                print('test_iter={},eta={},wer={},cer={}'.format(i_iter,eta,np.array(wer).mean(),np.array(cer).mean()))
                print(''.join(101 *'-'))

        return (np.array(loss_list).mean(), np.array(wer).mean(), np.array(cer).mean())

def train(model, net):
    if (opt.is_wbs):
        print('creating lm')
        wbs = init_word_beam_search()
        print('lm done')

    dataset = MyDataset(opt.video_path,
        opt.anno_path,
        opt.train_list,
        opt.vid_padding,
        opt.txt_padding,
        'train')

    loader = dataset2dataloader(dataset)
    optimizer = optim.Adam(model.parameters(),
                lr = opt.base_lr,
                weight_decay = 0.,
                amsgrad = True)

    print('num_train_data:{}'.format(len(dataset.data)))
    crit = nn.CTCLoss(zero_infinity=True)
    tic = time.time()

    train_wer = []
    for epoch in range(opt.max_epoch):
        for (i_iter, input) in enumerate(loader):
            model.train()
            vid = input.get('vid').cuda()
            txt = input.get('txt').cuda()
            vid_len = input.get('vid_len').cuda()
            txt_len = input.get('txt_len').cuda()

            optimizer.zero_grad()
            y = net(vid)
            
            loss = crit(y.transpose(0, 1).log_softmax(-1), txt, vid_len.view(-1), txt_len.view(-1))
            loss.backward()

            if loss == np.inf or loss == np.nan:
                print (y.transpose(0, 1).log_softmax(-1).shape)
                print (txt.shape)

            if(opt.is_optimize):
                optimizer.step()

            tot_iter = i_iter + epoch*len(loader)

            if (opt.is_wbs):
                pred_txt = ctc_decode2(y.permute(1, 0, 2).roll(-1, 2).softmax(-1).detach().cpu().numpy(), wbs)
            else:
                pred_txt = ctc_decode(y)
            
            truth_txt = [MyDataset.arr2txt(txt[_], start=1) for _ in range(txt.size(0))]
            train_wer.extend(MyDataset.wer(pred_txt, truth_txt))

            if(tot_iter % opt.display == 0):
                v = 1.0*(time.time()-tic)/(tot_iter+1)
                eta = (len(loader)-i_iter)*v/3600.0

                writer.add_scalar('train loss', loss, tot_iter)
                writer.add_scalar('train wer', np.array(train_wer).mean(), tot_iter)
                print(''.join(101*'-'))
                print('{:<75}|{:>75}'.format('predict', 'truth'))
                print(''.join(101*'-'))

                for (predict, truth) in list(zip(pred_txt, truth_txt))[:3]:
                    print('{:<75}|{:>75}'.format(predict, truth))
                print(''.join(101*'-'))
                print('epoch={},tot_iter={},eta={},loss={},train_wer={}'.format(epoch, tot_iter, eta, loss, np.array(train_wer).mean()))
                print(''.join(101*'-'))

            if(tot_iter % opt.test_step == 0):
                (loss, wer, cer) = test(model, net)
                print('i_iter={},lr={},loss={},wer={},cer={}'
                    .format(tot_iter,show_lr(optimizer),loss,wer,cer))
                writer.add_scalar('val loss', loss, tot_iter)
                writer.add_scalar('wer', wer, tot_iter)
                writer.add_scalar('cer', cer, tot_iter)
                savename = '{}_loss_{}_wer_{}_cer_{}.pt'.format(opt.save_prefix, loss, wer, cer)
                (path, name) = os.path.split(savename)
                if(not os.path.exists(path)): os.makedirs(path)
                torch.save(model.state_dict(), savename)
                if(not opt.is_optimize):
                    exit()

def init_word_beam_search():
    corpus = codecs.open('LRS3_corpus.txt', 'r', 'utf8').read()
    chars = codecs.open('chars.txt', 'r', 'utf8').read()
    word_chars = codecs.open('wordChars.txt', 'r', 'utf8').read()
    beam_width = 25
    lm_type = opt.wbs_type
    lm_smoothing = 0.0
    return WordBeamSearch(beam_width, lm_type, lm_smoothing, corpus.encode('utf8'), chars.encode('utf8'), word_chars.encode('utf8'))

if(__name__ == '__main__'):

    print("Loading options...")
    model = LipNet()
    model = model.cuda()
    net = nn.DataParallel(model).cuda()

    if(hasattr(opt, 'weights')):
        pretrained_dict = torch.load(opt.weights)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}
        missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict.keys()]
        print('loaded params/tot params:{}/{}'.format(len(pretrained_dict),len(model_dict)))
        print('miss matched params:{}'.format(missed_params))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    torch.manual_seed(opt.random_seed)
    torch.cuda.manual_seed_all(opt.random_seed)
    train(model, net)
    # results = test(model, net)

    # print('done')
    # print('loss: %f, wer: %f, cer: %f' % (results[0].mean(), results[1].mean(), results[2].mean()))
