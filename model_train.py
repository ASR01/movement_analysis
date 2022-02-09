import os
from tkinter.tix import NoteBook
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import model, device
import torch
import pickle
import copy
from tqdm import tqdm
from data_preparation import FramesDataset


with open('./data_red/dataloaders.pkl', 'rb') as p_file:
    dl = pickle.load(p_file)

train_dl = dl['train_dl']
test_dl = dl['test_dl']



epochs = 10
loss_func = torch.nn.CrossEntropyLoss(reduction="sum")
optimizer = optim.Adam(model.parameters(), lr=3e-5)
lr_scheduler = ReduceLROnPlateau(optimizer, mode='min',factor=0.5, patience=5,verbose=1)


from tqdm import tqdm

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_model(model, dl):

	train_dl= dl['train_dl']
	test_dl=dl['test_dl']
	#print(len(train_dl), len(test_dl))

	# path2weights=params["path2weights"]
	# sanity_check=params["sanity_check"]
    
	loss_history={
        "train": [],
        "val": [],
    }
    
	acc_history={
        "train": [],
        "val": [],
    }
    
	best_loss=float('inf')
    
	for epoch in range(epochs):
        
		current_lr=get_lr(optimizer)
        
		print('Epoch {}/{}, current lr={}'.format(epoch, epochs - 1, current_lr))
        
		model.train()
        
		epoch_loss_tr=0.0
		epoch_acc_tr=0.0
		len_tr_dl = len(train_dl.dataset)
		loop = tqdm(train_dl)  
		for i, (x, y) in enumerate(loop):
			x=x.to(device)
			y=y.to(device)
			output=model(x)
			#loss_b,metric_b=loss_batch(loss_func, output, yb, opt)
			loss = loss_func(output, y)
			with torch.no_grad():
				pred = output.argmax(dim=1, keepdim=True)
				acc_train=pred.eq(y.view_as(pred)).sum().item()
				#print(acc_train, pred.tolist(), y.tolist())			
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
    		
			loss_tr = loss.item()
			epoch_loss_tr += loss_tr
			epoch_acc_tr += acc_train

		#print(epoch_loss_tr, epoch_acc_tr, len_tr_dl)
		loss=epoch_loss_tr/float(len_tr_dl)
		acc=epoch_acc_tr/float(len_tr_dl)

		loop.set_description(f"Epoch [{epoch}/{epochs}]")
		loop.set_postfix(loss=epoch_loss_tr, acc=epoch_acc_tr)


		loss_history["train"].append(loss)
		acc_history["train"].append(acc)
		#print(loss_history, metric_history)    
	        
		model.eval()
		with torch.no_grad():
			epoch_val_loss = 0.0
			epoch_acc_val=0.0
			len_val_dl = len(test_dl.dataset)
			loop_val = tqdm(test_dl)  
			for i, (x, y) in enumerate(loop_val):
				x=x.to(device)
				y=y.to(device)
				output=model(x)
				loss = loss_func(output, y)
				pred = output.argmax(dim=1, keepdim=True)
				acc_val=pred.eq(y.view_as(pred)).sum().item()
				#print(acc_val, pred.tolist(), y.tolist())				
				loss_vb = loss.item()
				epoch_val_loss+=loss_vb
				epoch_acc_val+=acc_val
			loss_val=epoch_val_loss/float(len_val_dl)
			acc_val=epoch_acc_val/float(len_val_dl)
			loop_val.set_description(f"Epoch [{epoch}/{epochs}]")
			loop_val.set_postfix(loss=loss_val, acc=acc_val)
       

		if loss_val < best_loss:
			best_loss = loss_val
			torch.save(model.state_dict(), './model/state_dict.pt')
			print("Model weights actualised")
        	
		loss_history["val"].append(loss_val)
		acc_history["val"].append(acc_val)
 
        

		print("train loss: %.6f, test loss: %.6f, accuracy: %.2f" %(loss,loss_val,100*acc_val))
		print("-"*10) 
	# model.load_state_dict(best_model_wts)
            		
	return model, loss_history, acc_history
  



model,loss_hist,metric_hist = train_model(model, dl)