# deep-content-user

***An implementation of "Deep Content-User Embedding Model for Music Recommendation, [arxiv](https://arxiv.org/abs/1807.06786)"***

----------------------------------

## Requirements

- tensorflow-gpu==1.4.0
- keras==2.0.8

These requirements can be easily installed by:
	`pip install -r requirements.txt`

## Scripts

- __data_generator.py__: The base script that contains batch data generator and valid set loader.
- __load_label.py__: The base script for metadata loading.
- __mp3s_to_mel.py__: Convert audio to mel-spectrogram.
- __utils.py__: Contain functions for evaluation.
- __model.py__: Basic and multi models.
- __train.py__: Module for training the recommendation models.
- __encoding.py__: Contains script for extracting embedding vector given the trained model weight.
- __evaluation.py__: embedding evaluation script for recommendation experiment.
- __tagging.py__: Module for training the tagging experiment.

## Usage

Here are examples of how to run the code. (To run 1. and 2., you need MSD audio files and its related metadata from [Echonest-TasteProfile-DataLoader](https://github.com/kyungyunlee/Echonest-TasteProfile-DataLoader), [MSD_split](https://github.com/jongpillee/music_dataset_split/tree/master/MSD_split))
1. `python mp3s_to_mel.py` 
2. `python train.py basic --margin 0.2 --N-negs 20`
3. `python encoding.py basic ./models/model_basic_20_0.20/weights.555-6.84.h5`
4. `python evaluation.py basic`
5. `python tagging.py basic`

----------------------------------

## Reference

[1] [Deep Content-User Embedding Model for Music Recommendation](https://arxiv.org/abs/1807.06786), Jongpil Lee, Kyungyun Lee, Jiyoung Park, Jangyeon Park, and Juhan Nam, arxiv, 2018
