from __future__ import print_function
import numpy as np
import random
import os
import argparse

from model import *
from load_label import *
from keras.models import Model

def main(args):

	# load models
	model_str = '(args, inference=True)'
	if args.model == 'basic':
		audio_model_1, user_model_1 = eval('model_' + args.model + model_str)
		audio_model = [audio_model_1]
		user_model = [user_model_1]
	elif args.model == 'multi':
		audio_model_1,audio_model_2,audio_model_3,audio_model_4,audio_model_5,user_model_1,user_model_2,user_model_3,user_model_4,user_model_5 = eval('model_' + args.model + model_str)
		audio_model = [audio_model_1,audio_model_2,audio_model_3,audio_model_4,audio_model_5]
		user_model = [user_model_1,user_model_2,user_model_3,user_model_4,user_model_5]
		
	save_path = './embeddings/%s/' % args.model

	# user embedding
	for model_iter in range(len(user_model)):
		user_embedding = user_model[model_iter].predict(np.arange(args.num_user))
		user_save_str = 'user_embedding_%d.npy' % (model_iter+1)
		np.save(save_path + user_save_str, user_embedding)
	print('user embedding saved')

	# load label
	sorted_coo_train, sorted_coo_valid, songs, user_to_item_train, user_to_item_valid, all_items, D7id_to_path, Tid_to_D7id, Sid_to_Tid, item_to_user_train, train_idx, valid_idx, test_idx = load_label(args)

	# item embedding
	for model_iter in range(len(audio_model)):
		item_embedding = np.zeros((args.num_song,args.dim_embedding))
		for iter in range(len(songs)):
			try:
			
				file_path = args.feature_path + D7id_to_path[Tid_to_D7id[Sid_to_Tid[songs[iter]]]].replace('.mp3','.npy')
				tmp = np.load(file_path)
				tmp = tmp.T

				tmp -= args.mel_mean
				tmp /= args.mel_std
		
				# segmentation
				input_seg = []
				num_seg = int(tmp.shape[0]/args.num_frame)
				for iter2 in range(num_seg):
					input_seg.append(tmp[iter2*args.num_frame:(iter2+1)*args.num_frame,:])
	
				input_seg = np.array(input_seg)
				predicted = audio_model[model_iter].predict(input_seg)
				item_embedding[iter] = np.mean(predicted,axis=0)

			except Exception:
				continue

			if np.remainder(iter,1000) == 0:
				print(iter)
		print(iter+1)

		item_save_str = 'item_embedding_%d.npy' % (model_iter+1)
		np.save(save_path + item_save_str,item_embedding)
	print('item embedding saved')


if __name__ == '__main__':	

	# options 
	parser = argparse.ArgumentParser(description='feature vector extraction')
	parser.add_argument('model', type=str, default='basic', help='choose between basic model and multi model')
	parser.add_argument('weight_name', type=str, help='weight path')
	parser.add_argument('--N-negs', type=int, default=20, help='negative sampling size')
	parser.add_argument('--margin', type=float, default=0.2, help='margin value for hinge loss')
	parser.add_argument('--dim-embedding', type=int, default=256, help='feature vector dimension')
	parser.add_argument('--num-frame', type=int, default=130, help='frame size of input')
	parser.add_argument('--feature-path', type=str, default='/home1/irteam/users/jongpil/data/msd/mels/', help='mel-spectrogram path')
	parser.add_argument('--mel-mean', type=float, default=0.2262, help='mean value calculated from training set')
	parser.add_argument('--mel-std', type=float, default=0.2579, help='std value calculated from training set')
	parser.add_argument('--num-user', type=int, default=20000, help='the number of users')
	parser.add_argument('--num-song', type=int, default=10000, help='the number of items')
	parser.add_argument('--melBins', type=int, default=128, help='mel bin size')
	parser.add_argument('--lr', type=float, default=0.2, help='learning rate')
	parser.add_argument('--lrdecay', type=float, default=1e-6, help='learning rate decaying')
	args = parser.parse_args()

	main(args)

