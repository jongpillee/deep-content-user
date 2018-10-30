from __future__ import print_function
import os.path
import numpy as np
import librosa
import cPickle as cP

fftsize = 1024
window = 1024
hop = 512
melBin = 128

label_path = './50tagLabels/'

id7d_to_path = cP.load(open(label_path + '7D_id_to_path.pkl','r'))
idmsd_to_id7d = cP.load(open(label_path + 'MSD_id_to_7D_id.pkl','r'))

all_list = idmsd_to_id7d.keys()

load_path = 'MSD mp3s path'
save_path = 'MSD melspec save path'

for iter in range(0,len(all_list)):

	try:
		file_name = load_path + id7d_to_path[idmsd_to_id7d[all_list[iter]]]
		save_name = save_path + id7d_to_path[idmsd_to_id7d[all_list[iter]]].replace('.mp3','.npy')

		if not os.path.exists(os.path.dirname(save_name)):
			os.makedirs(os.path.dirname(save_name))

		if os.path.isfile(save_name) == 1:
			print(iter, save_name + '_file_already_extracted!')
			continue

		y,sr = librosa.load(file_name,sr=22050)
		S = librosa.core.stft(y,n_fft=fftsize,hop_length=hop,win_length=window)
		X = np.abs(S)

		mel_basis = librosa.filters.mel(sr,n_fft=fftsize,n_mels=melBin)
	
		mel_S = np.dot(mel_basis,X)

		mel_S = np.log10(1+10*mel_S)
		mel_S = mel_S.astype(np.float32)

		mel_S = mel_S[:,:1291]

		print(iter,mel_S.shape,save_name)
		np.save(save_name,mel_S)

	except Exception:
		continue






