import streamlit as st
import cv2
import tempfile
import os
from inference import predict 
import time

st.title('Movement classificator')

st.markdown('''
            Just upload a short video an it can be classified in the following six categories:
            
| Climbing | Dribbling | Golf  |
| Bike Riding  | Bow Shooting  | Baseball       |
            
''')

def save_uploadedfile(uploadedfile):
    with open(os.path.join("./temp",'temp_file.avi'),"wb") as f:
        f.write(uploadedfile.getbuffer())
        

videoloc = './temp/temp_file.avi'

f = st.file_uploader("Upload file")
suc = st.empty()
col1, col2 = st.columns(2)
with col1:
    start = st.button('Start')
    stvideo = st.empty()

with col2:
	# pr = st.button('Predict')
	response = st.empty()

if f != None:
	save_uploadedfile(f)
	tfile = tempfile.NamedTemporaryFile(delete=False)
	tfile.write(f.read())
	vf = cv2.VideoCapture(tfile.name)
	


if start == 1:
	video = vf
	video.set(cv2.CAP_PROP_FPS, 25)



	while True:
		success, image = video.read()
		if not success:
			break
		stvideo.image(image, channels="BGR")
		time.sleep(0.01)
	response.text('making prediction....')        
	pred, label = predict('temp/temp_file.avi')
	print(pred, label)
	str = (f'The activity class detected is:  *{label}* ')
	response.markdown(str)
