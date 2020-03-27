# This is the implementation of our paper in ISBI 2020.
# N. Zhang, et al, A Spatially Constrained Deep Convolutional Neural Network for Nerve Fiber Segmentation in Corneal Confocal Microscopic Images Using Inaccurate Annotations
# Please cite our paper if you find this code helpful.
# Sigma value in spatially_constrained_loss() and the weights for {'balance-cross-entropy':0.7, 'sc-loss':0.3} in line 71 are hyperparameters, which are application dependent.
# Contact: xin.chen@nottingham.ac.uk

import os
import glob
import shutil
import argparse
import numpy as np 
import scipy.io as sio
import tensorflow as tf 

from utils import util as U
from core.data_processor import SimpleImageProcessor
from core.data_provider import DataProvider
from core.learning_rate import StepDecayLearningRate
from core.trainer_tf import Trainer
from models.model import SimpleTFModel

from nets_tf.unet2d import UNet2D

parser = argparse.ArgumentParser()
parser.add_argument('-ep', '--epochs', type=int, default=200, help='number of epochs')
parser.add_argument('-bs', '--batch_size', type=int, default=4, help='batch size')
parser.add_argument('-mbs', '--minibatch_size', type=int, default=None, help='mini-batch size')
parser.add_argument('-ebs', '--eval_batch_size', type=int, default=1, help='mini-batch size')
parser.add_argument('-ef', '--eval_frequency', type=int, default=1, help='frequency of evaluation within training')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001, help='learning rate')
parser.add_argument('-out', '--output_path', type=str, default='results/CCMOutput/')
args = parser.parse_args()

output_path = args.output_path

# load data path list
data_path = 'D:/Projects/ISBI2020/CCM/'
train_set = glob.glob(data_path + '/train/*org.tif')
valid_set = glob.glob(data_path + '/validate/*org.tif')[:5]
test_set = glob.glob(data_path + '/test/*org.tif')[:5]

# set key for input images and labels
org_suffix = 'org'
lab_suffix = 'lab'

# init processor
pre = {org_suffix: ['zero-mean', ('channelcheck', 1)],
       lab_suffix: [('one-hot', [0,1]), ('channelcheck', 2)]}
processor = SimpleImageProcessor(pre=pre)

# If is_save_temp = True, a temp folder will be created in default temp folder of system. 
# Auto-remove will be applied when the code normal exit.
# Otherwise, user need to delete the temp folder manual. (System will do it?)
train_provider = DataProvider(train_set, [org_suffix, lab_suffix],
                        is_pre_load=False,
                        is_save_temp=False,   
                        is_shuffle=True,  
                        processor=processor)

validation_provider = DataProvider(valid_set, [org_suffix, lab_suffix],
                        is_pre_load=False,
                        is_save_temp=False,
                        processor=processor)

# init network
unet = UNet2D(n_class=2, n_layer=5, root_filters=16, use_bn=True)

# build model
# loss_function should like {'name': weight}
# weight_function should like {'name': {'alpha': 1, 'beta':2}} or ['name'] / ('name') / {'name'}
model = SimpleTFModel(unet, org_suffix, lab_suffix, dropout=0, loss_function={'balance-cross-entropy':0.9, 'sc-loss':0.1})

# set learning rate with step decay, every [decay_step] epoch, learning rate = [learning rate * decay_rate]
lr = StepDecayLearningRate(learning_rate=args.learning_rate, 
                           decay_step=100,
                           decay_rate=1,
                           data_size=train_provider.size,
                           batch_size=args.batch_size)

# init optimizer with learning rate [lr]
optimizer = tf.keras.optimizers.Adam(lr)

# init trainer
trainer = Trainer(model)

# start training
result = trainer.train(train_provider, validation_provider,
                       epochs=args.epochs,
                       batch_size=args.batch_size,
                       mini_batch_size=args.minibatch_size,
                       output_path=output_path,
                       optimizer=optimizer,
                       learning_rate=lr,
                       eval_frequency=args.eval_frequency,
                       is_save_train_imgs=False,
                       is_save_valid_imgs=True,
                       is_rebuilt_path=True)

# evaluate test data
test_provider = DataProvider(test_set, [org_suffix, lab_suffix],
                        is_pre_load=False,
                        processor=processor)

# evaluation and save
trainer.restore(output_path + '/ckpt/final')
eval_dcit = trainer.eval(test_provider, batch_size=args.eval_batch_size)
with open(output_path + '/test_eval.txt', 'a+') as f:
    f.write('final:' + U.dict_to_str(eval_dcit) + '\n')
sio.savemat(args.output_path + '/final_results.mat', eval_dcit)

trainer.restore(output_path + '/ckpt/best')
eval_dcit = trainer.eval(test_provider, batch_size=args.eval_batch_size)
with open(output_path + '/test_eval.txt', 'a+') as f:
    f.write('Best :' + U.dict_to_str(eval_dcit) + '\n')
sio.savemat(args.output_path + '/best_results.mat', eval_dcit)
print()
