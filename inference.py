import torch
import torchvision
from torchvision import transforms
from preprocessing import get_frames
from PIL import Image
import pickle

def prepare_frames(frames):

    trf = transforms.Compose([
                transforms.Resize((112,112)),
                transforms.ToTensor(),
                transforms.Normalize([0.43216, 0.394666, 0.37645], [0.22803, 0.22145, 0.216989])]) 

    frames_array = []
    for frame in frames:
        frame = Image.fromarray(frame)
        frame_trf = trf(frame)
        frames_array.append(frame_trf)
    frames_tensor = torch.stack(frames_array)    

    frames_tensor = torch.transpose(frames_tensor, 1, 0)
    frames_tensor = frames_tensor.unsqueeze(0)

    return frames_tensor


def main():
	pass

def predict(video):
	with open('./data_red/labels_dict.pkl', 'rb') as p_file:
		labels_dict = pickle.load(p_file)


	num_classes = 6
	model = torchvision.models.video.r3d_18(pretrained=True, progress=False)
	num_features = model.fc.in_features
	model.fc = torch.nn.Linear(num_features, num_classes)
	model.eval()

	device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

	weights = "./model/state_dict_90.pt"
	model.load_state_dict(torch.load(weights, map_location=device))
	model.to(device)


	frames, video_length = get_frames(video, n_frames=16)


###### Prepare frames

	trf = transforms.Compose([
                transforms.Resize((112,112)),
                transforms.ToTensor(),
                transforms.Normalize([0.43216, 0.394666, 0.37645], [0.22803, 0.22145, 0.216989])]) 

	frames_array = []
	for frame in frames:
		frame = Image.fromarray(frame)
		frame_trf = trf(frame)
		frames_array.append(frame_trf)
	frames_tensor = torch.stack(frames_array)    
	frames_tensor = torch.transpose(frames_tensor, 1, 0)
	frames_tensor = frames_tensor.unsqueeze(0)

	with torch.no_grad():
		out = model(frames_tensor.to(device)).cpu()
		#print(out.shape)
		pred = torch.argmax(out).item()
		#print(pred)
		label = labels_dict[pred]
	return pred, label 
   
if __name__ == '__main__':
	video = './data_red/hmdb/golf/9_Iron_From_160_Yards_golf_f_cm_np1_ri_goo_0.avi'
	
	pred, label = predict(video)
	print(pred,label)