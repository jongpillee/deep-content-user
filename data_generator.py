from __future__ import print_function
import numpy as np
import random
import threading

class threadsafe_iter:
	def __init__(self,it):
		self.it = it
		self.lock = threading.Lock()

	def __iter__(self):
		return self

	def next(self):
		with self.lock:
			return self.it.next()

def threadsafe_generator(f):
	def g(*a, **kw):
		return threadsafe_iter(f(*a,**kw))
	return g

@threadsafe_generator
def train_generator(args, sorted_coo, songs, user_to_item, all_items, D7id_to_path, Tid_to_D7id, Sid_to_Tid, steps_per_epoch, train_idx, valid_idx, test_idx, item_to_user):

	random.shuffle(sorted_coo)
	#random.shuffle(train_idx)
	while True:
		
		for batch_iter in range(0, steps_per_epoch*args.batch_size, args.batch_size):

			# initializing
			x_train_batch = []
			y_train_batch = []

			col_anchor_user = []
			col_pos_item = []
			col_neg_items = [[] for j in range(args.N_negs)]
	
			for item_idx,item_iter in enumerate(range(batch_iter,batch_iter+args.batch_size)):

	
	
				#item_to_user, user_to_item, co_list, sorted_coo
				user_idx = sorted_coo[item_iter][1]
				item_idx = sorted_coo[item_iter][0]
				#theuserlist = item_to_user[train_idx[item_iter]]
				#user_idx = random.choice(theuserlist)

				# anchor user
				anchor_user = user_idx

				# load pos item
				file_path = args.feature_path + D7id_to_path[Tid_to_D7id[Sid_to_Tid[songs[item_idx]]]].replace('.mp3','.npy')
				#file_path = args.feature_path + D7id_to_path[Tid_to_D7id[Sid_to_Tid[songs[train_idx[item_iter]]]]].replace('.mp3','.npy')

				tmp = np.load(file_path)
				tmp = tmp.T
						
				# repeat when its too short
				if tmp.shape[0] < args.num_frame:
					tmp = np.tile(tmp,(100,1))
					
				tmp -= args.mel_mean
				tmp /= args.mel_std


				start = random.randint(0,tmp.shape[0]-args.num_frame)
				pos_item = tmp[start:start+args.num_frame,:]

				neg_item_list = user_to_item[user_idx]
				#neg_item_list = list(set(all_items) - set(neg_item_list))
				neg_item_list = list(set(all_items) - set(neg_item_list) - set(valid_idx) - set(test_idx))


				# load neg items
				tmp_neg_items = []
				for neg_iter in range(args.N_negs):
						
					neg_idx = random.choice(neg_item_list)

					file_path = args.feature_path + D7id_to_path[Tid_to_D7id[Sid_to_Tid[songs[neg_idx]]]].replace('.mp3','.npy')
				
					tmp = np.load(file_path)
					tmp = tmp.T
						
					# repeat when its too short
					if tmp.shape[0] < args.num_frame:
						tmp = np.tile(tmp,(100,1))
					
					tmp -= args.mel_mean
					tmp /= args.mel_std


					start = random.randint(0,tmp.shape[0]-args.num_frame)

					tmp_neg_items.append(tmp[start:start+args.num_frame,:])

				for neg_iter in range(args.N_negs):
					col_neg_items[neg_iter].append(tmp_neg_items[neg_iter])

				col_anchor_user.append(anchor_user)
				col_pos_item.append(pos_item)
	
				label = np.zeros((args.N_negs+1))
				label[0] = 1
				y_train_batch.append(list(label))	


			col_anchor_user = np.array(col_anchor_user)
			col_pos_item = np.array(col_pos_item)
			for j in range(args.N_negs):
				col_neg_items[j] = np.array(col_neg_items[j])


			x_train_batch = [col_anchor_user, col_pos_item] + [col_neg_items[j] for j in range(args.N_negs)]

			if args.model == 'basic':
				y_train_batch = np.array(y_train_batch)
			elif args.model == 'multi':
				y_train_batch = [np.array(y_train_batch),np.array(y_train_batch),np.array(y_train_batch),np.array(y_train_batch),np.array(y_train_batch)]

			yield x_train_batch, y_train_batch


def load_valid(args, sorted_coo, songs, user_to_item, all_items, D7id_to_path, Tid_to_D7id, Sid_to_Tid, train_idx, valid_idx, test_idx, item_to_user):

	# load valid sets
	x_valid = []
	y_valid = []

	random.shuffle(sorted_coo)
	len_sparse = len(sorted_coo)

	# load 10 times valid set more randomly
	col_anchor_user = []
	col_pos_item = []
	col_neg_items = [[] for j in range(args.N_negs)]
	for valid_iter in range(1):

		for iter in range(int(len(valid_idx))):

			user_idx = sorted_coo[iter][1]
			item_idx = sorted_coo[iter][0]
			#theuserlist = item_to_user[valid_idx[iter]]
			#user_idx = random.choice(theuserlist)

			# anchor user
			anchor_user = user_idx

			# load pos item
			file_path = args.feature_path + D7id_to_path[Tid_to_D7id[Sid_to_Tid[songs[item_idx]]]].replace('.mp3','.npy')
			#file_path = args.feature_path + D7id_to_path[Tid_to_D7id[Sid_to_Tid[songs[valid_idx[iter]]]]].replace('.mp3','.npy')

			#tmp = all_audio[file_path]
			tmp = np.load(file_path)
			tmp = tmp.T
						
			# repeat when its too short
			if tmp.shape[0] < args.num_frame:
				tmp = np.tile(tmp,(100,1))
					
			tmp -= args.mel_mean
			tmp /= args.mel_std


			start = random.randint(0,tmp.shape[0]-args.num_frame)
			pos_item = tmp[start:start+args.num_frame,:]

			neg_item_list = user_to_item[user_idx]
			#neg_item_list = list(set(all_items) - set(neg_item_list))
			neg_item_list = list(set(all_items) - set(neg_item_list) - set(train_idx) - set(test_idx))

			# load neg items
			tmp_neg_items = []
			for neg_iter in range(args.N_negs):

				neg_idx = random.choice(neg_item_list)
				#print('neg_idx',neg_idx)

				file_path = args.feature_path + D7id_to_path[Tid_to_D7id[Sid_to_Tid[songs[neg_idx]]]].replace('.mp3','.npy')
				
				#tmp = all_audio[file_path]
				tmp = np.load(file_path)
				tmp = tmp.T
						
				# repeat when its too short
				if tmp.shape[0] < args.num_frame:
					tmp = np.tile(tmp,(100,1))
					
				tmp -= args.mel_mean
				tmp /= args.mel_std


				start = random.randint(0,tmp.shape[0]-args.num_frame)
					
					
				tmp_neg_items.append(tmp[start:start+args.num_frame,:])

			for neg_iter in range(args.N_negs):
				col_neg_items[neg_iter].append(tmp_neg_items[neg_iter])

			col_anchor_user.append(anchor_user)
			col_pos_item.append(pos_item)
					
			label = np.zeros((args.N_negs+1))
			label[0] = 1
			y_valid.append(list(label))	


			if np.remainder(iter,1000) == 0:
				print(iter)
		print(iter+1)
	print(len(x_valid))

	col_anchor_user = np.array(col_anchor_user)
	col_pos_item = np.array(col_pos_item)
	for j in range(args.N_negs):
		col_neg_items[j] = np.array(col_neg_items[j])

	x_valid = [col_anchor_user, col_pos_item] + [col_neg_items[j] for j in range(args.N_negs)]
	if args.model == 'basic':
		y_valid = np.array(y_valid)
	elif args.model == 'multi':
		y_valid = [np.array(y_valid),np.array(y_valid),np.array(y_valid),np.array(y_valid),np.array(y_valid)]

	return x_valid, y_valid


