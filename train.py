from __future__ import print_function
import numpy as np
import cPickle as cP
import random
import os
import argparse

from model import *
from data_generator import *
from load_label import *

from keras.optimizers import SGD
from keras.callbacks import Callback,EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from keras import backend as K
from keras.models import Model

def hinge_loss(y_true,y_pred):
	# hinge loss
	y_pos = y_pred[:,:1]
	y_neg = y_pred[:,1:]
	loss = K.sum(K.maximum(0., args.margin - y_pos + y_neg))
	return loss
			
def main(args):

	
	# build model
	model = eval('model_' + args.model + '(args)')

	# model compile
	sgd = SGD(lr=args.lr,decay=args.lrdecay,momentum=0.9,nesterov=True)
	if args.model == 'basic':
		model.compile(optimizer=sgd,loss=hinge_loss,metrics=['accuracy'])
	elif args.model == 'multi':
		model.compile(optimizer=sgd,loss={'output_1': hinge_loss, 'output_2': hinge_loss, 'output_3': hinge_loss, 'output_4': hinge_loss, 'output_5': hinge_loss},
			loss_weights={'output_1':1.,'output_2':1.,'output_3':1.,'output_4':1.,'output_5':1.},metrics=['accuracy'])
	model.summary()

	# load label data
	sorted_coo_train, sorted_coo_valid, songs, user_to_item_train, user_to_item_valid, all_items, D7id_to_path, Tid_to_D7id, Sid_to_Tid, item_to_user, train_idx, valid_idx, test_idx = load_label(args)
	len_sparse = len(sorted_coo_train)

	# load valid data
	x_valid, y_valid = load_valid(args, sorted_coo_valid, songs, user_to_item_valid, all_items, D7id_to_path, Tid_to_D7id, Sid_to_Tid, train_idx, valid_idx, test_idx, item_to_user)

	# make weight directory
	steps_per_epoch = int(len_sparse/args.batch_size/10) # randomly chosen epoch
	weight_name = './models/model_%s_%d_%.2f/weights.{epoch:02d}-{val_loss:.2f}.h5' % (args.model,args.N_negs,args.margin)
	if not os.path.exists(os.path.dirname(weight_name)):
		os.makedirs(os.path.dirname(weight_name))

	# callbacks
	#callbacks = [EarlyStopping(monitor='val_loss',patience=4,verbose=1,mode='auto'),
	#				ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=2,verbose=1,mode='auto',min_lr=min_lr),
	#				ModelCheckpoint(monitor='val_loss',filepath=weight_name,verbose=0,save_best_only=False,mode='auto',period=N)]
	callbacks = [ModelCheckpoint(monitor='val_loss',filepath=weight_name,verbose=0,save_best_only=False,mode='auto',period=args.N)]


	# run model
	model.fit_generator(generator=train_generator(args, sorted_coo_train, songs, user_to_item_train, all_items, D7id_to_path, Tid_to_D7id, Sid_to_Tid, steps_per_epoch, train_idx, valid_idx, test_idx, item_to_user),
					steps_per_epoch=steps_per_epoch,
					workers=args.workers,
					use_multiprocessing=True,
					max_queue_size=1,
					epochs=args.epochs,
					verbose=1,
					callbacks=callbacks,
					validation_data=(x_valid,y_valid))
	print('training done!')



if __name__ == '__main__':

	# options
	parser = argparse.ArgumentParser(description='deep content-user embedding model')
	parser.add_argument('model', type=str, default='basic', help='choose between basic model and multi model')
	parser.add_argument('--N-negs', type=int, default=20, help='negative sampling size')
	parser.add_argument('--margin', type=float, default=0.2, help='margin value for hinge loss')
	parser.add_argument('--dim-embedding', type=int, default=256, help='feature vector dimension')
	parser.add_argument('--N', type=int, default=2, help='save weight every N epochs')
	parser.add_argument('--lr', type=float, default=0.2, help='learning rate')
	parser.add_argument('--lrdecay', type=float, default=1e-6, help='learning rate decaying')
	parser.add_argument('--min-lr', type=float, default=0.00000016, help='minimum learning rate')
	parser.add_argument('--epochs', type=int, default=10000, help='epochs')
	parser.add_argument('--batch_size', type=int, default=10, help='batch size')
	parser.add_argument('--workers', type=int, default=40, help='the number of generators to be used')
	parser.add_argument('--melBins', type=int, default=128, help='mel bin size')
	parser.add_argument('--num-frame', type=int, default=130, help='frame size of input')
	parser.add_argument('--feature-path', type=str, default='/home1/irteam/users/jongpil/data/msd/mels/', help='mel-spectrogram path')
	parser.add_argument('--mel-mean', type=float, default=0.2262, help='mean value calculated from training set')
	parser.add_argument('--mel-std', type=float, default=0.2579, help='std value calculated from training set')
	parser.add_argument('--num-user', type=int, default=20000, help='the number of users')
	parser.add_argument('--num-song', type=int, default=10000, help='the number of items')
	args = parser.parse_args()

	main(args)

	
