import numpy as np
from matplotlib import pyplot as plt
import scipy
import pandas as pd
import tensorflow as tf
from tensorflow import keras     
from keras.optimizers import Adam
from sklearn.metrics import classification_report
import cv2           
from sklearn.metrics import precision_score, recall_score, f1_score


 def convert_video_to_frames(video_path): 

	# Path to video file 
	vid = cv2.VideoCapture(path) 

    # Get the width and height of the frames
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH));
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT));
    
	# Used as counter variable 
	frames_count = 130
    frames=[[[[0]*3]*height]*width]*130;
     pos=0
	while true: 

		# vidObj object calls read 
		# function extract frames 
		success, image = vidObj.read() 
        if success==0:
              break
          
		# Saves the frames with frame-count 
		frames[pos]=image 
        pos+=1

   return frames  


def detect_temporal_segments(predictions, threshold=0.5):
    temporal_segments = []
    current_segment_start = None
    for frame_index, frame_prediction in enumerate(predictions):
        if np.max(frame_prediction) > threshold:
            if current_segment_start is None:
                current_segment_start = frame_index
        elif current_segment_start is not None:
            temporal_segments.append((current_segment_start, frame_index - 1))
            current_segment_start = None
    if current_segment_start is not None:
        temporal_segments.append((current_segment_start, len(predictions) - 1))
return temporal_segments



train_data ='C:/Users/User/Downloads/train_data.zip';
test_data ='C:/Users/User/Downloads/test_data.zip'
# load training data
x_train = data['train_x']
y_train = data['train_y']
num_classes=20

y_train=keras.utils.to_categorical(y_train, num_classes)
# load test data
x_test = data['test_x']
y_test = data['test_y']
y_test=keras.utils.to_categorical(y_test, num_classes)
   
n=x_train.shape[0]          0 1 2 .....m-1
m=130   #num_frames
width=x_train.shape[1]
height=x_train.shape[2]
frames=[[[[[0]*3]*height]*width]*m]*n
   for i in range(0,n):
       frames[i]=convert_video_to_frame(train_data[i])
   
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
   
# normalize data
x_train /= 255
x_test /= 255

img_class=y_test.shape[1]  
     
cnn_model = keras.Sequential(
    [
       keras.layers.Input(shape=(m,width,height,3)),
       keras.layers.Conv2D(32,kernel_size=(m,3,3,3),activation="relu",padding='same'),
       keras.layers.MaxPooling2D(pad_size=(2,2,2)),
       keras.layers.Conv2D(64,kernel_size=(m,3,3,3),activation="relu",padding='same'),
       keras.layers.MaxPooling2D(pad_size=(2,2,2)),
       keras.layers.Flatten(),
    ]
  )

cnn_model.compile(loss="mean_squared_error",optimizer="adam",metrics=['accuracy'])
history = cnn_model.fit(x_train, y_train, batch_size=64, epochs=50, validation_split=0.15)
cnn_model.save('CNN_model_SAT_6.h5')  

features= cnn_model.predict(x_train)

MAX_SEQ_LENGTH = 8
NUM_FEATURES = features.shape[2]

lstm_model = keras.Sequential([
    Input(shape=(MAX_SEQ_LENGTH, NUM_FEATURES)),  # Input shape for the LSTM layer
    LSTM(32, activation='relu', return_sequences=True),
    LSTM(64, activation='relu', return_sequences=True),
    LSTM(128, activation='relu'),
    Dense(img_class, activation='softmax')
])

def prepare_seq_frames(x_train,y_train):
    
num_samples=(n*m)/MAX_SEQ_LENGTH

xx= np.zeros(shape=(num_samples, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")
ii=0
jj=0
for i range (0,n):
    for j in range(0,m):
        xx[ii][jj]=features[i][j]
        if(jj==MAX_SEQ_LENGTH-1): ii+=1;jj=0
        else: jj+=1 

    
# X_train shape: (num_samples, MAX_SEQ_LENGTH, NUM_FEATURES)
# y_train shape: (num_samples, num_classes)

yy=[[0]*num_classes]*num_samples
row=0
c=0
for i range(0,num_samples):
    row=i/num_samples
    for j in range(0,num_classes):
        if(y_train[row][j]==1):c=j;break
    yy[i][c]=1  
    
return xx,yy    

xx,yy=prepare_seq_frame(x_train,y_train)
        
lstm_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
lstm_model.fit(xx, yy, epochs=10, batch_size=32, validation_split=0.2)
lstm_model.save('LSTM_model_SAT_6.h5')

feature_mat=cnn_mode.predict(x_test)
xt,yt=prepare_seq_frame(x_test,y_test)

prediction=lstm_model.predict(xt)
pred_label = np.argmax(pred_label, axis=1)
temporal_segments = detect_temporal_segments(predictions, threshold=0.5)

precision = precision_score(y_test, predicted_label, average='weighted')
recall = recall_score(y_test, predicted_label, average='weighted')
accuracy = accuracy_score(y_true, predicted_classes)

print("Accuracy=%.4f \n precision=%.4f \n recall=%.4f",accuracy,precision,recall)      

# Accuracy= 0.8132
# Precision= 0.7793
# Recall= 0.8201



   
   
