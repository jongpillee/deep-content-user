from __future__ import print_function
import numpy as np

from keras.utils import to_categorical
from keras.optimizers import SGD
from keras.callbacks import Callback,EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from keras import backend as K

from keras.layers import Conv1D, MaxPool1D, BatchNormalization, GlobalAvgPool1D, Dense, Dropout, Activation, Reshape, Input, Concatenate, dot, Add, Flatten, concatenate, Embedding, add
from keras.models import Model

def model_basic(args, inference=False):

	pos_anchor = Input(shape = (1,))
	pos_item = Input(shape = (args.num_frame,args.melBins))
	neg_items = [Input(shape = (args.num_frame,args.melBins)) for j in range(args.N_negs)]

	# user model **one hot**
	user_dict = Embedding(args.num_user, 300, input_length=1)
	user_flat = Flatten()
	user_activ1 = Activation('relu')
	user_dense2 = Dense(300)
	user_activ2 = Activation('relu')	
	user_sem = Dense(args.dim_embedding,activation='linear')

	# anchor user
	anchor_user_dense1 =  user_dict(pos_anchor)
	anchor_user_flat = user_flat(anchor_user_dense1)
	anchor_user_activ1 = user_activ1(anchor_user_flat)
	anchor_user_dense2 = user_dense2(anchor_user_activ1)
	anchor_user_activ2 = user_activ2(anchor_user_dense2)
	anchor_user_sem = user_sem(anchor_user_activ2)

	# item model **audio**
	conv1 = Conv1D(128,4,padding='same',use_bias=True,kernel_initializer='he_uniform')
	activ1 = Activation('relu')
	MP1 = MaxPool1D(pool_size=4)	
	conv2 = Conv1D(args.dim_embedding,4,padding='same',use_bias=True,kernel_initializer='he_uniform')
	activ2 = Activation('relu')
	MP2 = MaxPool1D(pool_size=4)
	conv3 = Conv1D(args.dim_embedding,4,padding='same',use_bias=True,kernel_initializer='he_uniform')
	activ3 = Activation('relu')
	MP3 = MaxPool1D(pool_size=4)
	conv4 = Conv1D(args.dim_embedding,2,padding='same',use_bias=True,kernel_initializer='he_uniform')
	activ4 = Activation('relu')
	MP4 = MaxPool1D(pool_size=2)
	conv5 = Conv1D(args.dim_embedding,1,padding='same',use_bias=True,kernel_initializer='he_uniform')
	activ5 = Activation('relu')

	avg_pool = GlobalAvgPool1D()
	item_sem = Dense(args.dim_embedding,activation='linear')

	# pos item
	pos_item_conv1 = conv1(pos_item)
	pos_item_activ1 = activ1(pos_item_conv1)
	pos_item_MP1 = MP1(pos_item_activ1)
	pos_item_conv2 = conv2(pos_item_MP1)
	pos_item_activ2 = activ2(pos_item_conv2)
	pos_item_MP2 = MP2(pos_item_activ2)
	pos_item_conv3 = conv3(pos_item_MP2)
	pos_item_activ3 = activ3(pos_item_conv3)
	pos_item_MP3 = MP3(pos_item_activ3)
	pos_item_conv4 = conv4(pos_item_MP3)
	pos_item_activ4 = activ4(pos_item_conv4)
	pos_item_MP4 = MP4(pos_item_activ4)
	pos_item_conv5 = conv5(pos_item_MP4)
	pos_item_activ5 = activ5(pos_item_conv5)
	pos_item_avg = avg_pool(pos_item_activ5)
	pos_item_sem = item_sem(pos_item_avg)

	# neg items
	neg_item_conv1s = [conv1(neg_item) for neg_item in neg_items]
	neg_item_activ1s = [activ1(neg_item_bn1) for neg_item_bn1 in neg_item_conv1s]
	neg_item_MP1s = [MP1(neg_item_activ1) for neg_item_activ1 in neg_item_activ1s]
	neg_item_conv2s = [conv2(neg_item_MP1) for neg_item_MP1 in neg_item_MP1s]
	neg_item_activ2s = [activ2(neg_item_bn2) for neg_item_bn2 in neg_item_conv2s]
	neg_item_MP2s = [MP2(neg_item_activ2) for neg_item_activ2 in neg_item_activ2s]
	neg_item_conv3s = [conv3(neg_item_MP2) for neg_item_MP2 in neg_item_MP2s]
	neg_item_activ3s = [activ3(neg_item_bn3) for neg_item_bn3 in neg_item_conv3s]
	neg_item_MP3s = [MP3(neg_item_activ3) for neg_item_activ3 in neg_item_activ3s]
	neg_item_conv4s = [conv4(neg_item_MP3) for neg_item_MP3 in neg_item_MP3s]
	neg_item_activ4s = [activ4(neg_item_bn4) for neg_item_bn4 in neg_item_conv4s]
	neg_item_MP4s = [MP4(neg_item_activ4) for neg_item_activ4 in neg_item_activ4s]
	neg_item_conv5s = [conv5(neg_item_MP4) for neg_item_MP4 in neg_item_MP4s]
	neg_item_activ5s = [activ5(neg_item_bn5) for neg_item_bn5 in neg_item_conv5s]
	neg_item_avgs = [avg_pool(neg_item_activ5) for neg_item_activ5 in neg_item_activ5s]
	neg_item_sems = [item_sem(neg_item_avg) for neg_item_avg in neg_item_avgs]


	v_p = dot([anchor_user_sem, pos_item_sem], axes = 1, normalize = True)
	v_ns = [dot([anchor_user_sem, neg_item_sem], axes = 1, normalize = True) for neg_item_sem in neg_item_sems]

	prob = concatenate([v_p] + v_ns)

	# for hinge loss, use linear
	output = Activation('linear', name='output')(prob)

	if inference is False:
		model = Model(inputs = [pos_anchor, pos_item] + neg_items, outputs = output)
		return model
	else:
		model = Model(inputs = [pos_anchor, pos_item] + neg_items, outputs = output)
		model.load_weights(args.weight_name)
		print('weight loaded')

		# model compile
		sgd = SGD(lr=args.lr,decay=args.lrdecay,momentum=0.9,nesterov=True)
		model.compile(optimizer=sgd,loss='mean_squared_error',metrics=['accuracy'])
		model.summary()

		# audio models
		# -1 layer
		audio_model = Model(inputs = pos_item, outputs = pos_item_sem) 
		audio_model.compile(optimizer=sgd,loss='mean_squared_error',metrics=['accuracy'])
		audio_model.summary()

		# user models
		# -1 layer
		user_model = Model(inputs = pos_anchor, outputs = anchor_user_sem) 
		user_model.compile(optimizer=sgd,loss='mean_squared_error',metrics=['accuracy'])
		user_model.summary()

		return audio_model,user_model


def model_multi(args, inference=False):

	pos_anchor = Input(shape = (1,))
	pos_item = Input(shape = (args.num_frame,args.melBins))
	neg_items = [Input(shape = (args.num_frame,args.melBins)) for j in range(args.N_negs)]

	# user model **one hot**
	user_dict = Embedding(args.num_user, 300, input_length=1)
	user_flat = Flatten()
	user_activ1 = Activation('relu')

	user_dense2_1 = Dense(300)
	user_activ2_1 = Activation('relu')	
	user_sem_1 = Dense(args.dim_embedding,activation='linear')

	user_dense2_2 = Dense(300)
	user_activ2_2 = Activation('relu')	
	user_sem_2 = Dense(args.dim_embedding,activation='linear')

	user_dense2_3 = Dense(300)
	user_activ2_3 = Activation('relu')	
	user_sem_3 = Dense(args.dim_embedding,activation='linear')

	user_dense2_4 = Dense(300)
	user_activ2_4 = Activation('relu')	
	user_sem_4 = Dense(args.dim_embedding,activation='linear')

	user_dense2_5 = Dense(300)
	user_activ2_5 = Activation('relu')	
	user_sem_5 = Dense(args.dim_embedding,activation='linear')
	

	# anchor user
	anchor_user_dense1 =  user_dict(pos_anchor)
	anchor_user_flat = user_flat(anchor_user_dense1)
	anchor_user_activ1 = user_activ1(anchor_user_flat)

	anchor_user_dense2_1 = user_dense2_1(anchor_user_activ1)
	anchor_user_activ2_1 = user_activ2_1(anchor_user_dense2_1)
	anchor_user_sem_1 = user_sem_1(anchor_user_activ2_1)

	anchor_user_dense2_2 = user_dense2_2(anchor_user_activ1)
	anchor_user_activ2_2 = user_activ2_2(anchor_user_dense2_2)
	anchor_user_sem_2 = user_sem_2(anchor_user_activ2_2)

	anchor_user_dense2_3 = user_dense2_3(anchor_user_activ1)
	anchor_user_activ2_3 = user_activ2_3(anchor_user_dense2_3)
	anchor_user_sem_3 = user_sem_3(anchor_user_activ2_3)

	anchor_user_dense2_4 = user_dense2_4(anchor_user_activ1)
	anchor_user_activ2_4 = user_activ2_4(anchor_user_dense2_4)
	anchor_user_sem_4 = user_sem_4(anchor_user_activ2_4)

	anchor_user_dense2_5 = user_dense2_5(anchor_user_activ1)
	anchor_user_activ2_5 = user_activ2_5(anchor_user_dense2_5)
	anchor_user_sem_5 = user_sem_5(anchor_user_activ2_5)


	# item model **audio**
	conv1 = Conv1D(128,4,padding='same',use_bias=True,kernel_initializer='he_uniform')
	activ1 = Activation('relu')
	MP1 = MaxPool1D(pool_size=4)	
	conv2 = Conv1D(args.dim_embedding,4,padding='same',use_bias=True,kernel_initializer='he_uniform')
	activ2 = Activation('relu')
	MP2 = MaxPool1D(pool_size=4)
	conv3 = Conv1D(args.dim_embedding,4,padding='same',use_bias=True,kernel_initializer='he_uniform')
	activ3 = Activation('relu')
	MP3 = MaxPool1D(pool_size=4)
	conv4 = Conv1D(args.dim_embedding,2,padding='same',use_bias=True,kernel_initializer='he_uniform')
	activ4 = Activation('relu')
	MP4 = MaxPool1D(pool_size=2)
	conv5 = Conv1D(args.dim_embedding,1,padding='same',use_bias=True,kernel_initializer='he_uniform')
	activ5 = Activation('relu')

	avg_pool = GlobalAvgPool1D()
	item_sem = Dense(args.dim_embedding,activation='linear')

	# pos item
	pos_item_conv1 = conv1(pos_item)
	pos_item_activ1 = activ1(pos_item_conv1)
	pos_item_MP1 = MP1(pos_item_activ1)
	pos_item_conv2 = conv2(pos_item_MP1)
	pos_item_activ2 = activ2(pos_item_conv2)
	pos_item_MP2 = MP2(pos_item_activ2)
	pos_item_conv3 = conv3(pos_item_MP2)
	pos_item_activ3 = activ3(pos_item_conv3)
	pos_item_MP3 = MP3(pos_item_activ3)
	pos_item_conv4 = conv4(pos_item_MP3)
	pos_item_activ4 = activ4(pos_item_conv4)
	pos_item_MP4 = MP4(pos_item_activ4)
	pos_item_conv5 = conv5(pos_item_MP4)
	pos_item_activ5 = activ5(pos_item_conv5)
	pos_item_avg = avg_pool(pos_item_activ5)
	pos_item_sem = item_sem(pos_item_avg)

	# neg items
	neg_item_conv1s = [conv1(neg_item) for neg_item in neg_items]
	neg_item_activ1s = [activ1(neg_item_bn1) for neg_item_bn1 in neg_item_conv1s]
	neg_item_MP1s = [MP1(neg_item_activ1) for neg_item_activ1 in neg_item_activ1s]
	neg_item_conv2s = [conv2(neg_item_MP1) for neg_item_MP1 in neg_item_MP1s]
	neg_item_activ2s = [activ2(neg_item_bn2) for neg_item_bn2 in neg_item_conv2s]
	neg_item_MP2s = [MP2(neg_item_activ2) for neg_item_activ2 in neg_item_activ2s]
	neg_item_conv3s = [conv3(neg_item_MP2) for neg_item_MP2 in neg_item_MP2s]
	neg_item_activ3s = [activ3(neg_item_bn3) for neg_item_bn3 in neg_item_conv3s]
	neg_item_MP3s = [MP3(neg_item_activ3) for neg_item_activ3 in neg_item_activ3s]
	neg_item_conv4s = [conv4(neg_item_MP3) for neg_item_MP3 in neg_item_MP3s]
	neg_item_activ4s = [activ4(neg_item_bn4) for neg_item_bn4 in neg_item_conv4s]
	neg_item_MP4s = [MP4(neg_item_activ4) for neg_item_activ4 in neg_item_activ4s]
	neg_item_conv5s = [conv5(neg_item_MP4) for neg_item_MP4 in neg_item_MP4s]
	neg_item_activ5s = [activ5(neg_item_bn5) for neg_item_bn5 in neg_item_conv5s]
	neg_item_avgs = [avg_pool(neg_item_activ5) for neg_item_activ5 in neg_item_activ5s]
	neg_item_sems = [item_sem(neg_item_avg) for neg_item_avg in neg_item_avgs]

	pos_item_conv5_avg = avg_pool(pos_item_conv5)
	pos_item_conv4_avg = avg_pool(pos_item_conv4)
	pos_item_conv3_avg = avg_pool(pos_item_conv3)
	pos_item_conv2_avg = avg_pool(pos_item_conv2)

	neg_item_conv5_avgs = [avg_pool(neg_item_conv5) for neg_item_conv5 in neg_item_conv5s]
	neg_item_conv4_avgs = [avg_pool(neg_item_conv4) for neg_item_conv4 in neg_item_conv4s]
	neg_item_conv3_avgs = [avg_pool(neg_item_conv3) for neg_item_conv3 in neg_item_conv3s]
	neg_item_conv2_avgs = [avg_pool(neg_item_conv2) for neg_item_conv2 in neg_item_conv2s]


	v_p_1 = dot([anchor_user_sem_1, pos_item_sem], axes = 1, normalize = True)
	v_ns_1 = [dot([anchor_user_sem_1, neg_item_sem], axes = 1, normalize = True) for neg_item_sem in neg_item_sems]

	v_p_2 = dot([anchor_user_sem_2, pos_item_conv5_avg], axes = 1, normalize = True)
	v_ns_2 = [dot([anchor_user_sem_2, neg_item_conv5_avg], axes = 1, normalize = True) for neg_item_conv5_avg in neg_item_conv5_avgs]

	v_p_3 = dot([anchor_user_sem_3, pos_item_conv4_avg], axes = 1, normalize = True)
	v_ns_3 = [dot([anchor_user_sem_3, neg_item_conv4_avg], axes = 1, normalize = True) for neg_item_conv4_avg in neg_item_conv4_avgs]

	v_p_4 = dot([anchor_user_sem_4, pos_item_conv3_avg], axes = 1, normalize = True)
	v_ns_4 = [dot([anchor_user_sem_4, neg_item_conv3_avg], axes = 1, normalize = True) for neg_item_conv3_avg in neg_item_conv3_avgs]

	v_p_5 = dot([anchor_user_sem_5, pos_item_conv2_avg], axes = 1, normalize = True)
	v_ns_5 = [dot([anchor_user_sem_5, neg_item_conv2_avg], axes = 1, normalize = True) for neg_item_conv2_avg in neg_item_conv2_avgs]


	prob_1 = concatenate([v_p_1] + v_ns_1)
	prob_2 = concatenate([v_p_2] + v_ns_2)
	prob_3 = concatenate([v_p_3] + v_ns_3)
	prob_4 = concatenate([v_p_4] + v_ns_4)
	prob_5 = concatenate([v_p_5] + v_ns_5)

	# for hinge loss, use linear
	output_1 = Activation('linear', name='output_1')(prob_1)
	output_2 = Activation('linear', name='output_2')(prob_2)
	output_3 = Activation('linear', name='output_3')(prob_3)
	output_4 = Activation('linear', name='output_4')(prob_4)
	output_5 = Activation('linear', name='output_5')(prob_5)

	if inference is False:
		model = Model(inputs = [pos_anchor, pos_item] + neg_items, outputs = [output_1,output_2,output_3,output_4,output_5])
		return model
	else:
		model = Model(inputs = [pos_anchor, pos_item] + neg_items, outputs = [output_1,output_2,output_3,output_4,output_5])
		model.load_weights(args.weight_name)
		print('weight loaded')

		# model compile
		sgd = SGD(lr=args.lr,decay=args.lrdecay,momentum=0.9,nesterov=True)
		model.compile(optimizer=sgd,loss='mean_squared_error',metrics=['accuracy'])
		model.summary()

		# audio models
		# -1 layer
		audio_model_1 = Model(inputs = pos_item, outputs = pos_item_sem) 
		audio_model_1.compile(optimizer=sgd,loss='mean_squared_error',metrics=['accuracy'])
		audio_model_1.summary()

		# -2 layer
		audio_model_2 = Model(inputs = pos_item, outputs = pos_item_conv5_avg) 
		audio_model_2.compile(optimizer=sgd,loss='mean_squared_error',metrics=['accuracy'])
		audio_model_2.summary()

		# -3 layer
		audio_model_3 = Model(inputs = pos_item, outputs = pos_item_conv4_avg) 
		audio_model_3.compile(optimizer=sgd,loss='mean_squared_error',metrics=['accuracy'])
		audio_model_3.summary()

		# -4 layer
		audio_model_4 = Model(inputs = pos_item, outputs = pos_item_conv3_avg) 
		audio_model_4.compile(optimizer=sgd,loss='mean_squared_error',metrics=['accuracy'])
		audio_model_4.summary()

		# -5 layer
		audio_model_5 = Model(inputs = pos_item, outputs = pos_item_conv2_avg) 
		audio_model_5.compile(optimizer=sgd,loss='mean_squared_error',metrics=['accuracy'])
		audio_model_5.summary()

		# user models
		# -1 layer
		user_model_1 = Model(inputs = pos_anchor, outputs = anchor_user_sem_1) 
		user_model_1.compile(optimizer=sgd,loss='mean_squared_error',metrics=['accuracy'])
		user_model_1.summary()

		# -2 layer
		user_model_2 = Model(inputs = pos_anchor, outputs = anchor_user_sem_2) 
		user_model_2.compile(optimizer=sgd,loss='mean_squared_error',metrics=['accuracy'])
		user_model_2.summary()

		# -3 layer
		user_model_3 = Model(inputs = pos_anchor, outputs = anchor_user_sem_3) 
		user_model_3.compile(optimizer=sgd,loss='mean_squared_error',metrics=['accuracy'])
		user_model_3.summary()

		# -4 layer
		user_model_4 = Model(inputs = pos_anchor, outputs = anchor_user_sem_4) 
		user_model_4.compile(optimizer=sgd,loss='mean_squared_error',metrics=['accuracy'])
		user_model_4.summary()

		# -5 layer
		user_model_5 = Model(inputs = pos_anchor, outputs = anchor_user_sem_5) 
		user_model_5.compile(optimizer=sgd,loss='mean_squared_error',metrics=['accuracy'])
		user_model_5.summary()

		return audio_model_1,audio_model_2,audio_model_3,audio_model_4,audio_model_5,user_model_1,user_model_2,user_model_3,user_model_4,user_model_5


	
