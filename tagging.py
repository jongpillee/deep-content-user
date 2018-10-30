from __future__ import print_function
import numpy as np
import cPickle as cP
import os
import itertools
from utils import *

import argparse

from keras.optimizers import SGD
from keras.callbacks import Callback,EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from keras import backend as K

from keras.layers import Conv1D, MaxPool1D, BatchNormalization, GlobalAvgPool1D, Dense, Dropout, Activation, Reshape, Input, Concatenate, dot, Add, Flatten, concatenate
from keras.models import Model
from keras.regularizers import l2

def load_label():
	# data load
	Sid_to_Tid = cP.load(open('./data/echonest_id_to_MSD_id.pkl','rb'))
	D7id_to_path = cP.load(open('./data/7D_id_to_path.pkl','rb'))
	Tid_to_D7id = cP.load(open('./data/MSD_id_to_7D_id.pkl','rb'))
	Tid_to_tagGT = cP.load(open('./data/msd_id_to_tag_vector.cP','rb'))
	song_str = './data/subset_songs_20000_10000.npy'
	songs = np.load(song_str)
	return Sid_to_Tid, D7id_to_path, Tid_to_D7id, Tid_to_tagGT, songs

def load_embedding(args, model_iter, Sid_to_Tid, D7id_to_path, Tid_to_D7id, Tid_to_tagGT, songs):

	# load embeddings
	embedding_path = './embeddings/%s/' % args.model
	item_str = 'item_embedding_%d.npy' % (model_iter+1)
	item_factor = np.load(embedding_path + item_str)
	dim_embedding = item_factor.shape[1]

	# finding co - tagging, recommendation list
	Tid_songs = []
	for iter in range(len(songs)):
		Tid_songs.append(Sid_to_Tid[songs[iter]])

	tag_list = Tid_to_tagGT.keys()
	co_songs = list(set(tag_list) & set(Tid_songs))
	print(len(co_songs))

	# embeddings dictionary
	Tid_to_tag = {}
	for iter in range(len(songs)):
		Tid_to_tag[Tid_songs[iter]] = item_factor[iter,:]

	# split 7/1/2
	split_t = 7
	split_v = 8
	total = np.arange(len(co_songs))
	train_idx = []
	for iter in range(split_t):
		train_idx.append(np.where(total%10==iter)[0])
	train_idx = list(itertools.chain(*train_idx))
	valid_idx = np.where(total%10==split_t)[0]
	test_idx = []
	for iter in range(split_v,10):
		test_idx.append(np.where(total%10==iter)[0])
	test_idx = list(itertools.chain(*test_idx))
	print(len(train_idx),len(valid_idx),len(test_idx))

	# data split
	x_train_tag = []
	x_valid_tag = []
	x_test_tag = []
	y_train_tag = []
	y_valid_tag = []
	y_test_tag = []
	for iter in range(len(train_idx)):
		x_train_tag.append(Tid_to_tag[co_songs[train_idx[iter]]])
		y_train_tag.append(Tid_to_tagGT[co_songs[train_idx[iter]]])
	for iter in range(len(valid_idx)):
		x_valid_tag.append(Tid_to_tag[co_songs[valid_idx[iter]]])
		y_valid_tag.append(Tid_to_tagGT[co_songs[valid_idx[iter]]])
	for iter in range(len(test_idx)):
		x_test_tag.append(Tid_to_tag[co_songs[test_idx[iter]]])
		y_test_tag.append(Tid_to_tagGT[co_songs[test_idx[iter]]])
	x_train_tag = np.array(x_train_tag)
	x_valid_tag = np.array(x_valid_tag)
	x_test_tag = np.array(x_test_tag)
	y_train_tag = np.squeeze(np.array(y_train_tag))
	y_valid_tag = np.squeeze(np.array(y_valid_tag))
	y_test_tag = np.squeeze(np.array(y_test_tag))

	return x_train_tag,x_valid_tag,x_test_tag,y_train_tag,y_valid_tag,y_test_tag,dim_embedding

def model(args,dim_embedding):

	embedding = Input(shape = (dim_embedding,))

	dense1 = Dense(dim_embedding*2)(embedding)
	bn1 = BatchNormalization()(dense1)
	activ1 = Activation('relu')(bn1)
	dense2 = Dense(dim_embedding*2)(activ1)
	bn2 = BatchNormalization()(dense2)
	activ2 = Activation('relu')(bn2)
	
	output = Dense(args.num_tag,activation='sigmoid')(activ2)

	model = Model(inputs = embedding, outputs = output)
	return model

def main(args):

	if args.model == 'basic':
		Niter = 1
	elif args.model == 'multi':
		Niter = 5

	# load metadata
	Sid_to_Tid, D7id_to_path, Tid_to_D7id, Tid_to_tagGT, songs = load_label()

	# model iteration
	for model_iter in range(Niter):
	
		# load data
		x_train_tag,x_valid_tag,x_test_tag,y_train_tag,y_valid_tag,y_test_tag,dim_embedding = load_embedding(args,model_iter, Sid_to_Tid, D7id_to_path, Tid_to_D7id, Tid_to_tagGT, songs)
	
		# build model
		tag_model = model(args,dim_embedding)

		# model compile
		sgd = SGD(lr=args.lr,decay=args.lrdecay,momentum=0.9,nesterov=True)
		tag_model.compile(optimizer=sgd,loss='binary_crossentropy',metrics=['accuracy'])
		tag_model.summary()

		callbacks = [EarlyStopping(monitor='val_loss',patience=9,verbose=1,mode='auto'),
						ReduceLROnPlateau(monitor='val_loss',factor=0.2,patience=4,verbose=1,mode='auto',min_lr=args.min_lr)]

		# run model
		tag_model.fit(x_train_tag,y_train_tag,
					batch_size=args.batch_size,
					epochs=args.epochs,
					verbose=1,
					callbacks=callbacks,
					validation_data=(x_valid_tag,y_valid_tag))
		print('training done!')

		# evaluation
		pred_tag = tag_model.predict(x_test_tag)
		tag_auc, _ = eval_retrieval(pred_tag,y_test_tag)

		print1 = '%s model,item factor %d auc: %.4f' % (args.model,model_iter+1,tag_auc)
		print(print1)


if __name__ == '__main__':

	# options
	parser = argparse.ArgumentParser(description='tagging experiment')
	parser.add_argument('model', type=str, default='basic', help='choose between basic model and multi model')
	parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
	parser.add_argument('--lrdecay', type=float, default=1e-6, help='learning rate decaying')
	parser.add_argument('--min-lr', type=float, default=0.00000016, help='minimum learning rate')
	parser.add_argument('--epochs', type=int, default=1000, help='epochs')
	parser.add_argument('--batch-size', type=int, default=10, help='batch size')
	parser.add_argument('--num_tag', type=int, default=50, help='the number of tags')
	args = parser.parse_args()

	main(args)



	
