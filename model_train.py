import os
from tkinter.tix import NoteBook
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from model import model, device
import torch
import pickle
from a2_data_preparation import FramesDataset


with open('./data_red/dataloaders.pkl', 'rb') as p_file:
    dl = pickle.load(p_file)

train_dl = dl['train_dl']
test_dl = dl['test_dl']



epochs = 1
loss_func = torch.nn.CrossEntropyLoss(reduction="sum")
optimizer = optim.Adam(model.parameters(), lr=3e-5)
lr_scheduler = ReduceLROnPlateau(optimizer, mode='min',factor=0.5, patience=5,verbose=1)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_model(model, dl):

	train_dl= dl['train_dl']
	val_dl=dl['test_dl']

	# path2weights=params["path2weights"]
	# sanity_check=params["sanity_check"]
    
	loss_history={
        "train": [],
        "val": [],
    }
    
	metric_history={
        "train": [],
        "val": [],
    }
    
	# best_model_wts = copy.deepcopy(model.state_dict())
	# best_loss=float('inf')
    
	for epoch in range(epochs):
        
		current_lr=get_lr(optimizer)
        
		print('Epoch {}/{}, current lr={}'.format(epoch, epochs - 1, current_lr))
        
		model.train()
        
		epoch_loss=0.0
		epoch_metric=0.0
		len_data = len(train_dl)
		print(len(train_dl), len(test_dl))
		print(train_dl.dataset[0])
		for i, (x, y) in enumerate(train_dl):
			x=x.to(device)
			y=y.to(device)
			output=model(x)
			loss = loss_func(output, y)
			# with torch.no_grad():
			# 	pred = output.argmax(dim=1, keepdim=True)
			# 	metric_b=pred.eq(y.view_as(pred)).sum().item()
			
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
    		
			loss_b = loss.item()
			epoch_loss+=loss_b
			print(epoch_loss, i)
			epoch_metric = 0
			#epoch_metric+=metric_b

		loss=epoch_loss/float(len_data)
		metric=epoch_metric/float(len_data)
        
        #train_loss, train_metric=loss_epoch(model,loss_func,train_dl,sanity_check,optimizer)
        
		loss_history["train"].append(loss)
		metric_history["train"].append(metric)
        
		# model.eval()
		# with torch.no_grad():
		# 	val_loss, val_metric=loss_epoch(model,loss_func,val_dl,sanity_check)
        # if val_loss < best_loss:
        #     best_loss = val_loss
        #     best_model_wts = copy.deepcopy(model.state_dict())
        #     torch.save(model.state_dict(), path2weights)
        #     print("Copied best model weights!")
        
        # loss_history["val"].append(val_loss)
        # metric_history["val"].append(val_metric)
        
        # lr_scheduler.step(val_loss)
        # if current_lr != get_lr(opt):
        #     print("Loading best model weights!")
        #     model.load_state_dict(best_model_wts)
        

        # print("train loss: %.6f, dev loss: %.6f, accuracy: %.2f" %(train_loss,val_loss,100*val_metric))
        # print("-"*10) 
    # model.load_state_dict(best_model_wts)
        
	return model, loss_history, metric_history



model,loss_hist,metric_hist = train_model(model, dl)