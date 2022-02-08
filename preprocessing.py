import os
import cv2
import numpy as np

# This file has the objetive of getting for all the videos stored in data_red, to generate a group of frames that are going to be infused into the model.  

def get_frames(filename, n_frames= 3):
    frames = []
    video_capture = cv2.VideoCapture(filename)
    n_fr_video = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_list= np.linspace(0, n_fr_video-1, n_frames+1, dtype = int) #get 16 frame indexes evenly in the video
    #print(n_fr_video, frame_list)

    for fn in range(n_fr_video):
        success, frame = video_capture.read()
        if success is False:
            continue
        if (fn in frame_list):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
            frames.append(frame) 
    video_capture.release()
    return frames, n_fr_video

def save_images(frames, path4frame):
	for id, frame in enumerate(frames):
		#print(id)
		frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  
		path = os.path.join(path4frame, "frame_"+str(id)+".jpg")
		#print(path)
		cv2.imwrite(path, frame)

def main():

	path2data = "./data_red"
	video_folder = "hmdb"
	img_folder = "hmdb_img"
	path2cat = os.path.join(path2data, video_folder)
	listOfCategories = os.listdir(path2cat)
	print(listOfCategories, len(listOfCategories))



	for c in listOfCategories:
		print("category:", c)
		print(path2cat, c)
		path = os.path.join(path2cat, c)
		list_subf = os.listdir(path)
		print('videos:', len(list_subf))
		print('-'*50)




	video_ext = ".avi"
	num_frames = 32 # We select 32 because we can drop the uneven ones if we go to a 16 images model.
	for root, dirs, files in os.walk(path2cat, topdown=False):
		#print(root, dirs, files)
		for filename in files:
			#print(filename)
			if video_ext not in filename:
				continue
			path2vid = os.path.join(root, filename)
			# print(path2vid)
			frames, len_video = get_frames(path2vid, n_frames=num_frames)
			path2img = path2vid.replace(video_folder, img_folder)
			path2img = path2img.replace(video_ext, "")
			# print(path2img)
			os.makedirs(path2img, exist_ok= True)
			save_images(frames, path2img)



if __name__ == '__main__':
    main()