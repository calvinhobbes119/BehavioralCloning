## Project: Behavioral Cloning Project [![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Model Architecture
---
I used Nvidia's end-to-end Self-Driving Deep Learning network for the Behavioral Cloning project. The network has 5 convolutional layers and 3 fully connected (dense) layers. I inserted dropouts between the first two dense (fully-connected) layers for regularization with dropout probability of 0.3.

![network](https://github.com/calvinhobbes119/BehavioralCloning/blob/master/DriveNetwork.png) 

Training, Validation and Testing
---
I started out by splitting the included dataset  80/20 into training and validation sets. I used the center image as well as the left and right camera images with a correction of -/+ 0.1 respectively. I also flipped these three images horizontally (using the negative of the steering angle as the corresponding output) to augment my dataset. I used the Adam optimizer with a batch size of 32 samples, and training for 7 epochs. After observing the performance of the network I collected additional data on stretches of the track where the car was veering off course, and included them in my dataset. I repeated this procedure until the car was able to go completely around the track without going off course. My final training dataset was ~66K image samples, and my validation set was ~14K image samples.

[![Augmented Data Set 1](https://github.com/calvinhobbes119/BehavioralCloning/blob/master/Untitled.png)](https://youtu.be/RFD8soBKVxM)

Future improvements
---
I plan to experiment using more data from the challenge track so the car successfully completes the challenge course as well. Currently the car makes it way through roughly 10% of the challenge course before veering off track. The training and validation losses are still fairly high after 20 epochs of training, indicating that the network weights have not yet converged. I am currently testing by increasing the number of epochs as well as collecting more data on the challenge track.
