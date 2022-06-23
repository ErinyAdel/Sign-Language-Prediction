# Egyptian Sign Language Prediciton Using CNN And LSTM 
Hand gestures are one of the nonverbal communication modalities used in sign language. It is
most often used by deaf people who have hearing or speech impairments to communicate with
other deaf people or even with normal people, so a Deep Learning model is implemented to
predict the signs and translate them accurately and that is our mission to help them.
The model has been trained over a huge number of labelled videos and processed by
Convolutional and LSTM layers. The dynamic visuals were extracted as points in three
dimensions by MediaPipe.
After using these methodologies in our model, we got a very efficient model with 100% training
and validation accuracy, 99.98% in testing, and very accurate results in real-time testing.

## The main challenges facing the purpose of the application are:
- The speed of sign recognition
- The accuracy of detecting the sign gesture to its matched word
- Connecting the model with a working web application

## Implementation Details :
- First of all, the Data Set is manually collected. We gathered clips from certified
references on popular platforms like YouTube

![Youtube Header](images/Screenshot%202022-06-18%20020846.png "لغة الاشارة")
- We then trimmed the parts of needed exact gestures using media editing software like
Adobe Premiere.
![Adobe Premier](images/Screenshot%202022-06-18%20021157.png "Trimming")

- Then a Fix_Video() function is applied to obtain a fixed frame rate on all videos, the
chosen frame rate is 30 FPS. The function Removes excessive frames from videos that
exceeds the specified number of FPS, and it duplicates frames in videos that are short of
the required FPS.
```python
def fixVideo(frames,video_name,startFrames=0,endFrames=0,middleFrames=0):
```

- To be able to identify signs, a very large database is required. Better results are
guaranteed with a larger database (if the videos are correctly set). The video
augmentation has been done with Python. Then Augmentation on the videos were applied
in order to add to the training efficiency, a list of 9 different augmentations were added to
the videos
```python
import imgaug.augmenters as iaa
augs=[iaa.Rotate(5),iaa.Rotate(10),iaa.Rotate(15),
      iaa.Rotate(-5),iaa.Rotate(-10),iaa.Rotate(-15),
      iaa.ShearX(5),iaa.ShearX(10),iaa.ShearX(-5),
      iaa.ShearX(-10),iaa.ScaleY(1.1),iaa.ScaleY(0.9),
      iaa.TranslateX(px=5),iaa.TranslateY(px=5),
      iaa.Sequential([iaa.TranslateY(px=5),iaa.TranslateX(px=5)])]
for video in df["Video"]:
    video_file=vread(video)
    output=aug.augment_images(video_file)
    vwrite(f'{video.split(".")[0]}_filp.mp4',output)
```

- After that, features from videos are extracted using a tool called MediaPipe. Its function
is to adjust the points on the focused body and hands positions in order to take part to
train the model later.
>In MediaPipe, a section called Holistic MediaPipe is used which contains the two main models used for adjusting the points, Hands and Pose. MediaPipe Hands utilizes an ML pipeline consisting of multiple models working together: A palm detection model that operates on the full image and returns an oriented hand bounding box. A hand landmark model that operates on the cropped image region defined by the palm detector and returns high-fidelity 3D hand key points. MediaPipe Poses use the same mechanism for the rest of the body. The Hand model is fully used to detect the hands in the extracted videos, and the Pose model is used only on points 13,15 and 14,16.

![Mediapipe Hands](images/Screenshot%202022-06-18%20021946.png)
![Mediapipe Pose](images/Screenshot%202022-06-18%20022101.png)

## Running the model:
A detection confidence in the Holistic model is adjusted such that the tracking confidence
is determined to be equal to 0.001 to be able to detect anything and extract any feature.

Once the video augmentation was stored and ready, all videos were processed through
the Google MediaPipe hand tracking technology. In order to input the videos, a python
script was created. This python script searches for videos in a folder and sends the videos
to MediaPipe. The final modified version of MediaPipe is the one responsible for storing
the outputs in new folders.
It loops on each frame of the video, then process that frame and as a result it assigns Left
Hand Landmarks, Right Hand Landmarks, Pose Landmarks. It loops on those landmarks
and gets the specified points, all points for hands, and points (13,15 - 14,16) for Pose as
mentioned. If the visibility is greater 0.2, it appends them in array with its X,Y, and Z
coordinates. If the visibility is less than the assigned number, it appends it all with zeros.
The model applies the same mechanism for all landmarks as shown in the following

```python
def extract_keypoints(results):
    la = np.array([[res.x, res.y, res.z] if res.visibility > 0.2 else [0,0,0] for res in np.array(results.pose_landmarks.landmark)[[13,15]]]) if results.pose_landmarks else np.zeros((2,3))
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21,3))
    ra = np.array([[res.x, res.y, res.z] if res.visibility > 0.2 else [0,0,0] for res in np.array(results.pose_landmarks.landmark)[[14,16]]]) if results.pose_landmarks else np.zeros((2,3))
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21,3))
    return np.concatenate([la ,lh ,ra , rh])
```

The Resulted Shape was (16366, 30, 46, 3) which means the batch number of videos we
extracted from the beginning is 16366 videos that is 30 frames per second for each, and
the number of features by the Holistic model is 46 (21 points for Right Hand + 2 Points
for the Right Pose + 21 points for the Left Hand + 2 points for the Left Pose), and finally
the X,Y, and Z coordinates for every point.

## Training Stage:

The challenge of this stage is to implement a model of an effective use.
The objective is to use Long short-term memory neural networks (LSTM) which is a
Recurrent Neural Network.
The first conundrum is to match the parameters with LSTM as RNN deals with linear
data.

### First suggested solution implementation:
PCA is implemented to apply Dimensionality Reduction.
We used Incremental PCA such that it can take the 3 points (X, Y, and Z) and extract one
point out of them (Represent them in 1 point)
The, after Looping on all frames of all videos, we apply partial-fit to it and determine its
learned outcome.
```python
from sklearn.decomposition import IncrementalPCA
IPCA=IncrementalPCA(n_components=1)
for video in tqdm(X_train):
    for frame in video:
        IPCA.partial_fit(frame)

def return_pca(data):
    pca=[]
    for video in tqdm(data):
        frames=[]
        for frame in video:
            frames.append(IPCA.transform(frame))
        pca.append(np.array(frames))
    return np.array(pca).reshape((-1,30,46))

X_train_pca=return_pca(X_train)
X_test_pca=return_pca(X_test)
X_valid_pca=return_pca(X_valid)
```
The Result (Variance Ratio) was approximately equals 0.677 which means 33% of data
was lost.
Then PCA-Transform on the data was applied and returned the result.
The Output shape: (16366, 30, 46), 16366 videos with 30 FPS each, and 46 features.

We then Trained LSTM on it as shown in the following:
```python
model2=Sequential(name="LstmModel")
model2.add(LSTM(256,return_sequences=True,input_shape=X_train_lstm.shape[1:]))
model2.add(LSTM(128,dropout=0.3))
model2.add(Dense(100,activation="relu"))
model2.add(Dropout(0.2))
model2.add(Dense(128,activation="relu"))
model2.add(Dropout(0.2))
model2.add(Dense(np.unique(y).shape[0],activation="softmax"))
model2.compile(optimizer="nadam",loss="sparse_categorical_crossentropy",metrics=['accuracy'])
```
### Second suggested solution implementation:
Using Reshape instead of PCA such that we multiplied the 3 points (X, Y, and Z) by the
46 features. (3 * 46) as shown in the following:
```python
X_train_lstm=X_train.reshape((-1,30,3*46))
X_test_lstm=X_test.reshape((-1,30,3*46))
X_valid_lstm=X_valid.reshape((-1,30,3*46))
```
### Third suggested solution implementation: (Effective)
We used Time Distributed Layer such that it applies the same convolution layer to each
Timestep (Frames) independently.

**Advantages**: the filter/feature learned by the convolution layer for every frame will be the
same for the other ones as it learns accumulatively on all frames not start to learn on
every single frame each time

A Conv1D (Convolution with 1 dimension) is applied in the first stage, as a filter 3x1
(size: 3) is applied on the 3 points.

The first Result (Result 1) is that it learned 64 filters

Then MaxPooling is applied to distribute extracted features by picking the maximum
ones from the steps/frames (technically the half) that contain the learned filters.

Then Conv1D is used again such that filter 3x1 (size: 3) on the 3 points is applied and the
result was that it learned 96 filters

The same cycle of conv1D and MaxPooling is applied.

After that, A Global MaxPooling is applied to determine the important filters from the
128 ones across the final 11 features.

The Final shape: (none, 30, 128) and that fits with the LSTM method.

For connecting features to each other, a 2 layers LSTM is added and as a result it learned
90 features from each frame (Hidden state).

Return_sequence is applied to return the whole sequence in order to make the 45 units
learn from all the previous 90 features.

Then a dense layer is added with variant units (making the linear layers)

Then batch_normalization is added to normalize the output batch (Limiting numbers to
be between -1 and 1)

A dropout (dropping some units to focus on others) is added in order to avoid Overfitting

A dense layer then is added with activation softmax in accordance to the predicted words.
The last step was to Compile the model.
The steps are shown in the following:
```python
model=Sequential(name="CNNLSTM")
model.add(TimeDistributed(Conv1D(64,kernel_size=3,padding="same",activation="relu"),input_shape=X_train.shape[1:]))
model.add(TimeDistributed(MaxPool1D()))
model.add(TimeDistributed(Conv1D(96,kernel_size=3,padding="same",activation="relu")))
model.add(TimeDistributed(MaxPool1D()))
model.add(TimeDistributed(Conv1D(128,kernel_size=3,padding="same",activation="relu")))
model.add(TimeDistributed(GlobalMaxPool1D()))
model.add(LSTM(90,dropout=0.4,return_sequences=True))
model.add(LSTM(45,dropout=0.4))
model.add(Dense(100,activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(50,activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(np.unique(y).shape[0],activation="softmax"))
model.compile(optimizer="nadam",loss="sparse_categorical_crossentropy",metrics=['accuracy'])
```

### Effective Solution Preformance:
| |**Train**|**Validation**|**Test**|
| ------ | -------- | ------- | ------- |
| **Accuracy** | 0.999 | 1.000 | 0.997 |


## How To Run The Model:
- You can Access python Script in the `python` Folder **(Most Accurate Solution)** and install project dependencies using
 ```
 pip install opencv-python numpy mediapipe arabic_reshaper python-bidi pillow tensorflow

 ```
>then yo can run it with
```
python run.py (--input | -i) (0 `for webcam`| path/to/video )
```

- or you can run run the javascript version of the model by running the react application in the `client` folder
```
cd client
npm start 
```

- or you can use the hosted version on [Heroku](https://sign-language-pred.herokuapp.com/) 




