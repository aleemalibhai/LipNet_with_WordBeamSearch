gpu = '3'
random_seed = 0
data_type = 'unseen'
# video_path = '/scratch/aalibhai/lip/'
# train_list = f'data/{data_type}_train.txt'
# val_list = f'data/{data_type}_val.txt'
# anno_path = 'GRID_align_txt'
video_path = '/scratch/aalibhai/LRS3_lip/'
train_list = f'data/LRS3_train.txt'
val_list = f'data/LRS3_val.txt'
anno_path = '/scratch/aalibhai/pretrain'
vid_padding = 75
txt_padding = 200
batch_size = 16
base_lr = 2e-5
num_workers = 16
max_epoch = 10000
display = 10
test_step = 1000
save_prefix = f'weights/LipNet_{data_type}'
is_optimize = True
# Argmax, Words, NGrams, NGramsForecast, NGramsForecastAndSample
wbs_type = 'Argmax'
is_wbs = False

#weights = '/home/aalibhai/Thesis/LipNet-CTCWBS/pretrain/LipNet_unseen_loss_0.44562849402427673_wer_0.1332580699113564_cer_0.06796452465503355.pt'
