# Egyptian Sign Language Prediciton Using CNN And LSTM With MediaPipe
Hand gestures are one of the nonverbal communication modalities used in sign language. It is
most often used by deaf people who have hearing or speech impairments to communicate with
other deaf people or even with normal people, so a Deep Learning model is implemented to
predict the signs and translate them accurately and that is our mission to help them.
The model has been trained over a huge number of labelled videos and processed by
Convolutional and LSTM layers. The dynamic visuals were extracted as points in three
dimensions by MediaPipe.
After using these methodologies in our model, we got a very efficient model with 100% training
and validation accuracy, 99.98% in testing, and very accurate results in real-time testing.

## The Main Challenges Facing The Purpose of The Application
- The speed of the sign recognition
- The accuracy of detecting the sign gesture to its matched word
- Connecting the model with a working web application

## Implementation Details:

### 1. Collecting The Data
The dataset is manually collected. Clips are scraped from YouTube platform

![Youtube Header](images/youtube_header.png "بحث عن لغة الاشارة المصرية")
  
  
### 2. Data Preprocessing
- Trimming the parts of needed gestures using Adobe Premiere.

![Adobe Premier](images/premiere_interface.png "Trimming")
  
<br />

- Fixing the number of frames of all videos to be balanced data.
Then a Fix_Video() function is applied to obtain a fixed frame number on all videos, the
chosen frame number is 30 frame per video. The function Removes excessive frames from videos that
exceeds the specified number of FPS, and it duplicates frames in videos that are short of
the required FPS.
```python
def fixVideo(frames, video_name, startFrames=0, endFrames=0, middleFrames=0):
```

<br />

- Augmentation the dataset size to be able to identify signs. Better results are
guaranteed with a larger database (if the videos are correctly set). The video
augmentation has been done with Python. Then Augmentation on the videos were applied
in order to add to the training efficiency, a list of 9 different augmentations were added to
the videos, and applying these augmentations once for original data and another for flipped/mirrored data
```python
import imgaug.augmenters as iaa

augs=[iaa.Rotate(5),iaa.Rotate(10),iaa.Rotate(15),
      iaa.Rotate(-5),iaa.Rotate(-10),iaa.Rotate(-15),
      iaa.ShearX(5),iaa.ShearX(10),iaa.ShearX(-5),
      iaa.ShearX(-10),iaa.ScaleY(1.1),iaa.ScaleY(0.9),
      iaa.TranslateX(px=5),iaa.TranslateY(px=5),
      iaa.Sequential([iaa.TranslateY(px=5),iaa.TranslateX(px=5)])]

aug = iaa.Fliplr(1)

for video in df["Video"]:
    video_file = vread(video)
    output=aug.augment_images(video_file)
    vwrite(f'{video.split(".")[0]}_filp.mp4',output)
```

<br /> 

### 3. Feature Extraction 
Features are extracted from videos using a tool called MediaPipe. Its function
is to adjust the points on the focused body and hands positions in order to take part to
train the model later.
>In MediaPipe, a section called Holistic MediaPipe is used which contains the two main models used for adjusting the points, Hands and Pose. MediaPipe Hands utilizes an ML pipeline consisting of multiple models working together: A palm detection model that operates on the full image and returns an oriented hand bounding box. A hand landmark model that operates on the cropped image region defined by the palm detector and returns high-fidelity 3D hand key points. MediaPipe Poses use the same mechanism for the rest of the body. The Hand model is fully used to detect the hands (with all 21 landmarks for each hand), but the Pose model is used only on specific points to be more accurate in front of the camera, these landmarks are 13,15 and 14,16 which are for left and right wrist and elbow.

![Mediapipe Hands](images/hand_landmarks.png)
![Mediapipe Pose](images/pose_landmarks.png)

```python
def extract_keypoints(results):
    la = np.array([[res.x, res.y, res.z] if res.visibility > 0.2 else [0,0,0] for res in np.array(results.pose_landmarks.landmark)[[13,15]]]) if results.pose_landmarks else np.zeros((2,3))
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21,3))
    ra = np.array([[res.x, res.y, res.z] if res.visibility > 0.2 else [0,0,0] for res in np.array(results.pose_landmarks.landmark)[[14,16]]]) if results.pose_landmarks else np.zeros((2,3))
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21,3))
    return np.concatenate([la ,lh ,ra , rh])
```
>A detection confidence in the Holistic model is adjusted such that the tracking confidence
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

<br />

## 4. Building The Model:
The input shape from MediaPipe Holistic model (16366, 30, 46, 3) which means the batch number of videos we
extracted from the beginning is 16366 videos that is 30 frames for each video, and
the number of features by the Holistic model is 46 
(21 points for Right Hand + 2 Points for the Right Pose + 21 points for the Left Hand + 2 points for the Left Pose), 
and finally the X,Y, and Z coordinates for every point.

```python
model=Sequential(name="CNNLSTM")
model.add(TimeDistributed(Conv1D(64, kernel_size=3, padding="same", activation="relu"), input_shape=X_train.shape[1:]))
model.add(TimeDistributed(MaxPool1D()))
model.add(TimeDistributed(Conv1D(96, kernel_size=3, padding="same", activation="relu")))
model.add(TimeDistributed(MaxPool1D()))
model.add(TimeDistributed(Conv1D(128, kernel_size=3, padding="same", activation="relu")))
model.add(TimeDistributed(GlobalMaxPool1D()))
model.add(LSTM(90, dropout=0.4, return_sequences=True))
model.add(LSTM(45, dropout=0.4))
model.add(Dense(100, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(50, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(np.unique(y).shape[0],activation="softmax"))

model.compile(optimizer="nadam",loss="sparse_categorical_crossentropy",metrics=['accuracy'])
```

>We used Time Distributed Layer such that it applies the same convolution layer to each
Timestep (Frames) independently.

>**Advantages**: the filter/feature learned by the convolution layer for every frame will be the
same for the other ones as it learns accumulatively on all frames not start to learn on
every single frame each time

>A Conv1D (Convolution with 1 dimension) is applied in the first stage, as a filter 3x1
(size: 3) is applied on the 3 points.

>The first Result (Result 1) is that it learned 64 filters

>Then MaxPooling is applied to distribute extracted features by picking the maximum
ones from the steps/frames (technically the half) that contain the learned filters.

>Then Conv1D is used again such that filter 3x1 (size: 3) on the 3 points is applied and the
result was that it learned 96 filters

>The same cycle of conv1D and MaxPooling is applied.

>After that, A Global MaxPooling is applied to determine the important filters from the
128 ones across the final 11 features.

>The Final shape: (none, 30, 128) and that fits with the LSTM method.

>For connecting features to each other, a 2 layers LSTM is added and as a result it learned
90 features from each frame (Hidden state).

>Return_sequence is applied to return the whole sequence in order to make the 45 units
learn from all the previous 90 features.

>Then a dense layer is added with variant units (making the linear layers)

>Then batch_normalization is added to normalize the output batch (Limiting numbers to
be between -1 and 1)

>A dropout (dropping some units to focus on others) is added in order to avoid Overfitting

>A dense layer then is added with activation softmax in accordance to the predicted words.
The last step was to Compile the model.
The steps are shown in the following:

<br />

## 4. The Model Preformance
| |**Train**|**Validation**|**Test**|
| ------ | -------- | ------- | ------- |
| **Accuracy** | 0.999 | 1.000 | 0.997 |

<br />

## How To Run The Model:
- Install project dependencies using
 ```
 pip install opencv-python numpy mediapipe arabic_reshaper python-bidi pillow tensorflow
 ```
>Run it with
```
python run.py (--input | -i) (0 `for webcam`| path/to/video)
```

- Or run the javascript version of the model by running the react application in the `client` folder
```
cd client
npm start 
```

- Or use the hosted version on [Heroku](https://sign-language-pred.herokuapp.com/) 
