import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import SimpleRNN, Input, Dense
from sklearn.metrics import precision_score, recall_score, accuracy_score


def convert_video_to_frames(video_path, m): 
    # video capture object initialised using Path to video file video_path
    vid = cv2.VideoCapture(video_path) 

    # Get the width and height of the frames
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # List to store frames
    frames = []
    
    # Read frames until there are no more
    while ind < m: 
        # Read frame
        success, image = vid.read() 
        
        # Check if reading was successful,if it's unsuccessful,means no more frames are there
        if not success:
            break

        # Append frame to list
        frames.append(image)
        ind = ind+1
    
    height = frames[0].shape[0]
    width = frames[0].shape[1]
    while ind < m:
        # Pad with a blank frame
        frames.append(np.zeros((height, width, 3), dtype=np.uint8))
        ind += 1
    # Release video capture object
    vid.release()
    
    return frames  

def prepare_seq_frames(features, y_train):
   MAX_SEQ_LENGTH = 10
   n = features.shape[0]
   m = features.shape[1]
   NUM_FEATURES = features.shape[2]
   num_samples = (n*m)/MAX_SEQ_LENGTH
   xx = np.zeros(shape=(num_samples, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")
   ii=0
   jj=0
   for i range (0,n):
      for j in range(0,m):
        xx[ii][jj]=features[i][j]
        if(jj==MAX_SEQ_LENGTH-1): ii+=1;jj=0
        else: jj+=1 

    
   # X_train shape: (num_samples, MAX_SEQ_LENGTH, NUM_FEATURES)
   # y_train shape: (num_samples, num_classes)
   num_classes = 51
   yy=[[0]*num_classes]*num_samples
   row=0
   c=0
   for i range(0,num_samples):
      row=i/num_samples
      for j in range(0,num_classes):
          if(y_train[row][j]==1):c=j;break
      yy[i][c]=1  
    
   return xx,yy    

def prepare_train_data_frames():
    train_data_path = 'C:/Users/User/Downloads/hmdb51/train_data.zip'
    # load training data
    train_data_x = np.load(os.path.join(train_data_path, 'train_x.npy'))
    train_data_y = np.load(os.path.join(train_data_path, 'train_y.npy'))
    num_classes = 51  # Number of classes in HMDB51 dataset
    # Convert labels to one-hot encoding
    n = x_train.shape[0]
    m = 130  # num_frames
    x_train = [];
    for i in range(0, n):
        x_train.append(convert_video_to_frame(train_data_x[i], m))
        
    x_train = prepare_train_data_frames(train_data_x)
    x_train = x_train.astype('float32')
    x_train /= 255
    
    return x_train, train_data_y

def prepare_test_data_frames():
    test_data_path = 'C:/Users/User/Downloads/hmdb51/test_data.zip'
    # load training data
    test_data_x = np.load(os.path.join(test_data_path, 'test_x.npy'))
    test_data_y = np.load(os.path.join(test_data_path, 'test_y.npy'))
    num_classes = 51  # Number of classes in HMDB51 dataset

    n = x_test.shape[0]
    m = 130  # num_frames
    x_test= [];
    for i in range(0, n):
        x_test.append(convert_video_to_frame(test_data_x[i], m))

    x_test = prepare_train_data_frames(test_data_x)
    x_test = x_train.astype('float32')
    x_test /= 255

    return x_test, test_data_y

cnn_model = keras.Sequential(
    [
       keras.layers.Input(shape=(width,height,3)),
       keras.layers.Conv2D(32,kernel_size=(m,3,3,3),activation="relu",padding='same'),
       keras.layers.MaxPooling2D(pad_size=(2,2,2)),
       keras.layers.Conv2D(64,kernel_size=(m,3,3,3),activation="relu",padding='same'),
       keras.layers.MaxPooling2D(pad_size=(2,2,2)),
       keras.layers.Flatten()
    ]
  )
cnn_model_feature_extractor = keras.Model(inputs=cnn_model.input, outputs=cnn_model.layers[-3].output)

img_class = 51
rnn_model = keras.Sequential([
    Input(shape=(MAX_SEQ_LENGTH, NUM_FEATURES)),  # Input shape for the RNN layer
    SimpleRNN(32, activation='relu', return_sequences=True),
    SimpleRNN(64, activation='relu', return_sequences=True),
    SimpleRNN(128, activation='relu'),
    Dense(img_class, activation='softmax')
])

def fit_the_model():
    x_train = prepare_train_data_frames()  

    cnn_model.compile(loss="mean_squared_error",optimizer="adam",metrics=['accuracy'])
    history = cnn_model.fit(x_train, y_train, batch_size=64, epochs=50, validation_split=0.15)
    cnn_model.save('CNN_model_SAT_6.h5')  

    features = []
    for i in range(0,n):
        features.append(cnn_model_feature_extractor.predict(x_train[i]))

    xx,yy = prepare_seq_frames(features, y_train)
        
    rnn_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    rnn_model.fit(xx, yy, epochs=10, batch_size=32, validation_split=0.2)
    rnn_model.save('LSTM_model_SAT_6.h5')
   

def main():
    
    fit_the_model()
    x_test, y_test = prepare_test_data_frames()  
    features = []
    for i in range(0,n):
        features.append(cnn_model_feature_extractor.predict(x_test[i]))
        
    xt,yt = prepare_seq_frames(features, y_test)
    prediction = lstm_model.predict(xt)
    pred_label = np.argmax(prediction, axis=1)
    n = x_test.shape[0]
    m = x_test.shape[1]
    MAX_SEQ_LENGTH = 10
    tot_num_sequences = (n*m)/10
    one_num_sequences = m/10
    predicted_label = [];
    k=0
    output = []
    for i range(0,n):
        start_idx = i * one_num_sequences
        end_idx = (i + 1) * one_num_sequences
        # Extract labels for the current video segment
        video_segment_labels = pred_label[start_idx:end_idx]
        # Count occurrences of each label
        label_counts = np.bincount(video_segment_labels)
        # Get the label with the maximum count
        max_count_label = np.argmax(label_counts)
        # Assign the max count label to predicted_label
        predicted_label[i] = max_count_label
            
    precision = precision_score(y_test, predicted_label, average='weighted')
    recall = recall_score(y_test, predicted_label, average='weighted')
    accuracy = accuracy_score(y_true, predicted_classes)

    print("Accuracy=%.4f \n precision=%.4f \n recall=%.4f",accuracy,precision,recall)      

    # Accuracy= 0.8132
    # Precision= 0.7793
    # Recall= 0.8201



   
