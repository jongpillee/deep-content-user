from __future__ import print_function
import os
import numpy as np
import itertools
import argparse
from scipy import sparse
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity
from load_label import *

def main(args):

	embedding_path = './embeddings/%s/' % args.model

	if args.model == 'basic':
		Niter = 1
	elif args.model == 'multi':
		Niter = 5

	# load embeddings
	item_factors = []
	user_factors = []
	for model_iter in range(Niter):
		item_str = 'item_embedding_%d.npy' % (model_iter+1)
		user_str = 'user_embedding_%d.npy' % (model_iter+1)

		item_factor = np.load(embedding_path + item_str)
		user_factor = np.load(embedding_path + user_str)
		item_factors.append(item_factor)
		user_factors.append(user_factor)
		print(item_factor.shape,user_factor.shape)

	# reconstruct user listening history matrix from embeddings
	recon_mats = []
	for model_iter in range(Niter):
		recon_mat = cosine_similarity(item_factors[model_iter],user_factors[model_iter])
		recon_mats.append(recon_mat)

	# load original sparse matrix
	item_user_str = './data/song_user_matrix_20000_10000.npz'
	item_user_mat = sparse.load_npz(item_user_str)
	org_mat = item_user_mat.toarray()
	org_mat[org_mat>=1] = 1 # binary data
	print(org_mat.shape)

	# load split
	co_list,_,_,_,_ = get_co_list(args.feature_path, args.num_song)
	train_idx, valid_idx, test_idx = split(args.num_song,co_list)

	# get only train matrix (not true cold-start case)
	for model_iter in range(Niter):
		recon_mats[model_iter] = recon_mats[model_iter][train_idx,:]
	org_mat = org_mat[train_idx,:]
	print(recon_mats[0].shape,org_mat[0].shape)

	'''
	# get only test matrix (cold-start case)
	for model_iter in range(Niter):
		recon_mats[model_iter] = recon_mats[model_iter][test_idx,:]
	org_mat = org_mat[test_idx,:]
	print(recon_mats[0].shape,org_mat[0].shape)
	'''

	# calculate AUCs
	store_auc = [[] for x in range(Niter)]
	pop_auc = []
	pop_items = np.sum(org_mat,axis=1)

	for iter in range(args.num_user):

		recon_for_user = [recon_mats[x][:,iter] for x in range(Niter)]
		org_for_user = org_mat[:,iter]

		try:
			# AUC  
			recon_auc_for_user = [metrics.roc_auc_score(org_for_user,recon_for_user[x]) for x in range(Niter)]
			pop_auc_for_user = metrics.roc_auc_score(org_for_user,pop_items)
	
			for model_iter in range(Niter):
				store_auc[model_iter].append(recon_auc_for_user[model_iter])
			pop_auc.append(pop_auc_for_user)
					
			# prints 
			print_str = 'Iteration:' + str(iter) + '   '
			pop_str = 'Popularity auc: %.4f' % np.mean(np.asarray(pop_auc))
			store_strs = []
			for model_iter in range(Niter):
				store_str = '  predicted auc_%d: %.4f' % (model_iter+1, np.mean(np.asarray(store_auc[model_iter])))
				store_strs.append(store_str)

			print(print_str + pop_str, store_strs)
        
		except Exception:
			continue



if __name__ == '__main__':

	# options
	parser = argparse.ArgumentParser(description='evaluation')
	parser.add_argument('model', type=str, default='basic', help='choose between basic model and multi model')
	parser.add_argument('--num-user', type=int, default=20000, help='the number of users')
	parser.add_argument('--num-song', type=int, default=10000, help='the number of items')
	parser.add_argument('--feature-path', type=str, default='/home1/irteam/users/jongpil/data/msd/mels/', help='mel spectrogram path')
	args = parser.parse_args()

	main(args)



