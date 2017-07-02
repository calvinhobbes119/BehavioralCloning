 ## Project: Behavioral Cloning Project [![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Model Architecture
---
I used Nvidia's end-to-end Self-Driving Deep Learning network for the Behavioral Cloning project. The network has 5 convolutional layers and 3 fully connected (dense) layers. To prevent overfitting, as suggested by the project rubric, I inserted dropouts between the first two dense (fully-connected) layers for regularization with dropout probability of 0.3. Determining a good network architecture to solve a given problem appears to be more of an art than a science, and involves extensive experimentation. For this project, I relied on existing literature to find a good starting point for my network architecture. So I chose to start with the Nvidia architecutre, which has shown very promising results. It turned out that this architecture was sufficient for successfully achieving the objectives of this project, so I did not have to evaluate variations of the architecture. However, I did explore changing the dropout probability to 0.5 (the performance was worse with this on that dataset I had) before settling on a value of 0.3 which appeared to work better. It is possible that with a more complex problem (like on the challenge track) requiring more extensive training data, a dropout probability of 0.5 may work better to avoid overfitting, but this is still under investigating. As a preprocessing/normalization step, as recommended in the Project videos, I normalized the images so the pixel values are between [-0.5,+0.5], and cropped the images to exclude 70 pixel-rows from the top, and 25-pixel rows from the bottom of each image.

![network](https://github.com/calvinhobbes119/BehavioralCloning/blob/master/DriveNetwork.png) 

The following the output of Keras' model.summary() method. As the print out shows, this network has approx. 350K training parameters.

```text
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
lambda_1 (Lambda)                (None, 160, 320, 3)   0           lambda_input_1[0][0]
____________________________________________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 65, 320, 3)    0           lambda_1[0][0]
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 31, 158, 24)   1824        cropping2d_1[0][0]
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 31, 158, 24)   0           convolution2d_1[0][0]
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 14, 77, 36)    21636       activation_1[0][0]
____________________________________________________________________________________________________
activation_2 (Activation)        (None, 14, 77, 36)    0           convolution2d_2[0][0]
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 5, 37, 48)     43248       activation_2[0][0]
____________________________________________________________________________________________________
activation_3 (Activation)        (None, 5, 37, 48)     0           convolution2d_3[0][0]
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 3, 35, 64)     27712       activation_3[0][0]
____________________________________________________________________________________________________
activation_4 (Activation)        (None, 3, 35, 64)     0           convolution2d_4[0][0]
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 1, 33, 64)     36928       activation_4[0][0]
____________________________________________________________________________________________________
activation_5 (Activation)        (None, 1, 33, 64)     0           convolution2d_5[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 2112)          0           activation_5[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           211300      flatten_1[0][0]
____________________________________________________________________________________________________
activation_6 (Activation)        (None, 100)           0           dense_1[0][0]
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 100)           0           activation_6[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dropout_1[0][0]
____________________________________________________________________________________________________
activation_7 (Activation)        (None, 50)            0           dense_2[0][0]
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 50)            0           activation_7[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dropout_2[0][0]
____________________________________________________________________________________________________
activation_8 (Activation)        (None, 10)            0           dense_3[0][0]
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          activation_8[0][0]
====================================================================================================
Total params: 348,219
Trainable params: 348,219
Non-trainable params: 0
____________________________________________________________________________________________________
```
Training, Validation and Testing
---
I started out by splitting the dataset which was included with the Udacity project using an 80/20 ratio into training and validation sets. I used the center image as well as the left and right camera images from this dataset with a correction of -/+ 0.1 respectively. I initially tried a correction value of -/+ 0.2, but that did not work well. In order to augment my dataset, I also flipped these three images horizontally (using the negative of the steering angle as the corresponding training output). I used the Adam optimizer with a batch size of 32 samples, and experimented with training for upto 20 epochs. After observing the performance of the network I collected additional data on stretches of the track where the car was veering off course, and included the center/left/right images from this data as well in my training and validation. I've included a video of this dataset below where I drive over two particularly troublesome stretches of the track where the car repeatedly went off track.

[![Augmented Data Set 1](https://github.com/calvinhobbes119/BehavioralCloning/blob/master/Untitled.png)](https://youtu.be/RFD8soBKVxM)

In addition, I also included one additional dataset by going over the track for 3 full laps with smooth turns. With all of this data used for training and validation (again with an 80/20 split), the car was able to go completely around the track without going off course. My final training dataset was ~66K image samples, and my validation set was ~14K image samples. After observing the training and validation loss metrics, I settled on 7 epochs as the proper balance for the network weights to converge, without overfitting.

```text
62328/62328 [==============================] - 139s - loss: 0.0132 - val_loss: 0.0111
Epoch 2/20
62328/62328 [==============================] - 128s - loss: 0.0117 - val_loss: 0.0108
Epoch 3/20
62328/62328 [==============================] - 127s - loss: 0.0111 - val_loss: 0.0104
Epoch 4/20
62328/62328 [==============================] - 128s - loss: 0.0107 - val_loss: 0.0102
Epoch 5/20
62328/62328 [==============================] - 127s - loss: 0.0104 - val_loss: 0.0105
Epoch 6/20
62328/62328 [==============================] - 127s - loss: 0.0100 - val_loss: 0.0104
Epoch 7/20
62328/62328 [==============================] - 127s - loss: 0.0096 - val_loss: 0.0102
Epoch 8/20
62328/62328 [==============================] - 127s - loss: 0.0090 - val_loss: 0.0103
Epoch 9/20
62328/62328 [==============================] - 127s - loss: 0.0086 - val_loss: 0.0104
Epoch 10/20
62328/62328 [==============================] - 127s - loss: 0.0082 - val_loss: 0.0105
Epoch 11/20
62328/62328 [==============================] - 127s - loss: 0.0078 - val_loss: 0.0108
Epoch 12/20
62328/62328 [==============================] - 127s - loss: 0.0074 - val_loss: 0.0109
Epoch 13/20
62328/62328 [==============================] - 127s - loss: 0.0071 - val_loss: 0.0106
Epoch 14/20
62328/62328 [==============================] - 127s - loss: 0.0069 - val_loss: 0.0107
Epoch 15/20
62328/62328 [==============================] - 127s - loss: 0.0065 - val_loss: 0.0107
Epoch 16/20
62328/62328 [==============================] - 127s - loss: 0.0063 - val_loss: 0.0107
Epoch 17/20
62328/62328 [==============================] - 127s - loss: 0.0061 - val_loss: 0.0107
Epoch 18/20
62328/62328 [==============================] - 127s - loss: 0.0058 - val_loss: 0.0108
Epoch 19/20
62328/62328 [==============================] - 127s - loss: 0.0055 - val_loss: 0.0108
Epoch 20/20
62328/62328 [==============================] - 127s - loss: 0.0053 - val_loss: 0.0111
```
![network](https://github.com/calvinhobbes119/BehavioralCloning/blob/master/myfig.png) 
Future improvements
---
I plan to experiment using more data from the challenge track so the car successfully completes the challenge course as well. Currently the car makes it way through roughly 10% of the challenge course before veering off track. The training and validation losses are still fairly high after 20 epochs of training, indicating that the network weights have not yet converged. I am currently testing by increasing the number of epochs as well as collecting more data on the challenge track.
