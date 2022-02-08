import os
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, DataLoader, Subset
import glob
from PIL import Image
import torch
import numpy as np
import random
import torchvision.transforms as transforms
from model import collate_fn
import pickle

import matplotlib.pylab as plt


total_frames =16 #Images per video

class FramesDataset(Dataset):
	def __init__(self, ids, labels, labels_dict, transform = None):
		self.transform = transform
		self.ids = ids
		self.labels = labels
		self.labels_dict = labels_dict
  
	def __len__(self):
		return len(self.ids)

	def __getitem__(self, idx):
		path2imgs=glob.glob(self.ids[idx]+"/*.jpg")
		path2imgs = path2imgs[:total_frames]
		label = self.labels_dict[self.labels[idx]]

		frames = []
		for p2i in path2imgs:
			frame = Image.open(p2i)
			frames.append(frame)
		
		seed = np.random.randint(1e9)
		frames_tr = []
		for frame in frames:

			random.seed(seed)
			np.random.seed(seed)
			frame = self.transform(frame)
			frames_tr.append(frame)
		if len(frames_tr)>0:
			frames_tr = torch.stack(frames_tr)
		return frames_tr, label


def get_vids(path2img):
	categories = os.listdir(path2img)
	ids = []
	labels = []
	for c in categories:
		path2catg = os.path.join(path2img, c)
		listOfSubCats = os.listdir(path2catg)
		path2subCats= [os.path.join(path2catg,los) for los in listOfSubCats]
		ids.extend(path2subCats)
		labels.extend([c]*len(listOfSubCats))
	return ids, labels, categories 




def main():

################################ Prepare Data #################################
	path2data = "./data_red"
	sub_folder_jpg = "hmdb_img"
	path2img = os.path.join(path2data, sub_folder_jpg)


	img_folder, labels, cat = get_vids(path2img)

	labels_dict = {}
	i = 0
	for c in cat:
		labels_dict[i] = c
		i+=1

	with open('./data_red/labels_dict.pkl', 'wb') as p_file:
		pickle.dump(labels_dict, p_file)


################################ Datasets #################################


	split = StratifiedShuffleSplit(n_splits=2, test_size=0.1,	random_state=0)
	train_index, test_index = next(split.split(img_folder, labels))

################################ List od data #################################


	train_ids = [img_folder[i] for i in train_index]
	train_labels = [labels[i] for i in train_index]
	#print(len(train_ids), len(train_labels))

	test_ids = [img_folder[i] for i in test_index]
	test_labels = [labels[i] for i in test_index]
	#print(len(test_ids), len(test_labels))



################################ Generate TOrchvision Trf #################################

	train_trf = transforms.Compose([	
			transforms.Resize((112,112)),
			transforms.RandomHorizontalFlip(p=0.5),
			transforms.RandomAffine(degrees=0, translate=(0.1,0.1)),
			transforms.ToTensor(),
			transforms.Normalize([0.43216, 0.394666, 0.37645], [0.22803, 0.22145, 0.216989]),
			])
 
	test_trf = transforms.Compose([	
			transforms.Resize((112,112)),
			transforms.ToTensor(),
			transforms.Normalize([0.43216, 0.394666, 0.37645], [0.22803, 0.22145, 0.216989]),
			])


################################ Generate Datasets #################################


	train_ds = FramesDataset(ids= train_ids, labels= train_labels, labels_dict = labels_dict, transform= train_trf)
	print(len(train_ds))

	test_ds = FramesDataset(ids= test_ids, labels= test_labels, labels_dict = labels_dict, transform= test_trf)
	print(len(test_ds))

	# imgs, label = train_ds[1]
	# if len(imgs)>0:
	# 	#print(imgs.shape, label, torch.min(imgs), torch.max(imgs))
 

	batch_size = 16

	train_dl = DataLoader(train_ds, batch_size= batch_size, shuffle=True, collate_fn= collate_fn)
	test_dl = DataLoader(test_ds, batch_size= batch_size, shuffle=False, collate_fn= collate_fn)


	dl = {'train_dl' : train_dl, 
		'test_dl' : test_dl,
		'labels_dict' : labels_dict
	}

	
	with open('./data_red/dataloaders.pkl', 'wb') as p_file:
		pickle.dump(dl, p_file)




if __name__ == '__main__':
    main()