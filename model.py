import os
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import glob
import torch
import torchvision.models
import numpy as np
import random

num_classes = 6


def collate_fn(batch):
	imgs_batch, label_batch = list(zip(*batch))
	imgs_batch = [imgs for imgs in imgs_batch if len(imgs)>0]
	label_batch = [torch.tensor(lbl) for lbl, imgs in zip(label_batch, imgs_batch) if len(imgs)>0]
	imgs_tensor = torch.stack(imgs_batch)
	imgs_tensor = torch.transpose(imgs_tensor, 2, 1)
	labels_tensor = torch.stack(label_batch)
	return imgs_tensor,labels_tensor


model = torchvision.models.video.r3d_18(pretrained=True, progress=False)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, num_classes)



with torch.no_grad():
	x = torch.zeros(1, 3, 16, 112, 112)
	y= model(x)
	 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

#print(model)