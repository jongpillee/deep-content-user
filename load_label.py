from __future__ import print_function
import numpy as np
import cPickle as cP
import random
import itertools
import os
import threading
from scipy import sparse
import pandas as pd
from itertools import izip

def sort_coo(m):
	tuples = izip(m.row, m.col, m.data)
	return sorted(tuples, key=lambda x: (x[2]), reverse=True)

def load_label(args):


	song_user_str = './data/song_user_matrix_20000_10000.npz'
	song_user_csr = sparse.load_npz(song_user_str)
	song_user_coo = sparse.coo_matrix(song_user_csr)


	# find negative examples
	# (item, user)
	# first make user's item list (already composed)
	# except that randomly generate negative samples
	df = pd.DataFrame({'item':song_user_coo.row,'user':song_user_coo.col,'data':song_user_coo.data})


	# load co list
	co_list,Sid_to_Tid,D7id_to_path,Tid_to_D7id,songs =	get_co_list(args.feature_path,args.num_song)
	
	# load split
	train_idx, valid_idx, test_idx = split(args.num_song, co_list)
	print(df.shape)

	# filtering matrix
	df_train = df[df['item'].isin(train_idx)]
	print(df_train.shape)
	
	user_to_item_train = df_train.groupby('user')['item'].apply(list)
	item_to_user_train = df_train.groupby('item')['user'].apply(list)

	song_user_coo_train = sparse.coo_matrix(sparse.csr_matrix(((df_train.values[:,0].astype(int),(df_train.values[:,1],df_train.values[:,2])))))
	print(song_user_coo_train.shape)

	sorted_coo_train = sort_coo(song_user_coo_train)

	df_valid = df[df['item'].isin(valid_idx)]
	print(df_valid.shape)
	
	user_to_item_valid = df_valid.groupby('user')['item'].apply(list)
	item_to_user_valid = df_valid.groupby('item')['user'].apply(list)

	song_user_coo_valid = sparse.coo_matrix(sparse.csr_matrix(((df_valid.values[:,0].astype(int),(df_valid.values[:,1],df_valid.values[:,2])))))
	print(song_user_coo_valid.shape)

	sorted_coo_valid = sort_coo(song_user_coo_valid)


	all_items = co_list
	print('label loaded')


	return sorted_coo_train, sorted_coo_valid, songs, user_to_item_train, user_to_item_valid, all_items, D7id_to_path, Tid_to_D7id, Sid_to_Tid, item_to_user_train, train_idx, valid_idx, test_idx


def get_co_list(feature_path,num_song):
	# data load
	Sid_to_Tid = cP.load(open('./data/echonest_id_to_MSD_id.pkl','rb'))
	D7id_to_path = cP.load(open('./data/7D_id_to_path.pkl','rb'))
	Tid_to_D7id = cP.load(open('./data/MSD_id_to_7D_id.pkl','rb'))
	song_str = './data/subset_songs_20000_10000.npy'
	songs = np.load(song_str)

	# read files in audio subdirectories
	audio_list = []
	for path, subdirs, files in os.walk(feature_path):
		for name in files:
			if not name.startswith('.'):
				tmp = os.path.join(path,name)
				tmp = tmp.split('/')
				audio_list.append(tmp[8]+'/'+tmp[9]+'/'+tmp[10].replace('.npy','.mp3'))
	print(len(audio_list))

	path_to_D7id = dict(zip(D7id_to_path.values(), D7id_to_path.keys()))
	D7id_to_Tid = dict(zip(Tid_to_D7id.values(), Tid_to_D7id.keys()))
	Tid_to_Sid = dict(zip(Sid_to_Tid.values(), Sid_to_Tid.keys()))

	Sid_audio = []
	for iter in range(len(audio_list)):
		try:
			Sid_audio.append(Tid_to_Sid[D7id_to_Tid[path_to_D7id[audio_list[iter]]]])
		except Exception:
			continue

	# compare with songs
	idx_to_songs = dict(zip(np.arange(num_song), songs))
	songs_to_idx = dict(zip(idx_to_songs.values(), idx_to_songs.keys()))

	co_songs = list(set(Sid_audio) & set(songs))

	co_list = []
	for iter in range(len(co_songs)):
		co_list.append(songs_to_idx[co_songs[iter]])

	return co_list,Sid_to_Tid,D7id_to_path,Tid_to_D7id,songs 


def split(num_song,co_list):
	# train / valid / test split (7/1/2)
	split_t = 7
	split_v = 8
	total = np.arange(num_song)
	train_idx = []
	for iter in range(split_t):
		train_idx.append(np.where(total%10==iter)[0])
	train_idx = list(itertools.chain(*train_idx))
	valid_idx = np.where(total%10==split_v)[0]
	test_idx = []
	for iter in range(split_v,10):
		test_idx.append(np.where(total%10==iter)[0])
	test_idx = list(itertools.chain(*test_idx))
	print(len(train_idx),len(valid_idx),len(test_idx))

	train_idx = list(set(train_idx) & set(co_list))
	valid_idx = list(set(valid_idx) & set(co_list))
	test_idx = list(set(test_idx) & set(co_list))
	print(len(train_idx),len(valid_idx),len(test_idx))
	return train_idx, valid_idx, test_idx

