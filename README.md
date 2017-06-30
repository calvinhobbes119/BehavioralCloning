## Project: Behavioral Cloning Project [![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Model Architecture
---
I used Nvidia's end-to-end Self-Driving Deep Learning network for the Behavioral Cloning project. The network has 5 convolutional layers and 3 fully connected (dense) layers. I inserted dropouts between the first two dense (fully-connected) layers for regularization.

![network](https://github.com/calvinhobbes119/BehavioralCloning/blob/master/DriveNetwork.png) 

Training, Validation and Testing
---
I started out by splitting the included dataset  80/20 into training and validation sets. I used the Adam optimizer with a batch size of 32 samples, and training for 7 epochs. After observing the performance of the network I collected additional data on stretches of the track where the car was veering off course, and included them in my dataset. I repeated this procedure until the car was able to go completely around the track without going off course.

![drive video](https://github.com/calvinhobbes119/BehavioralCloning/blob/master/run1.mp4) 

[![Watch the video](https://img.youtube.com/vi/T-D1KVIuvjA/0.jpg)](https://github.com/calvinhobbes119/BehavioralCloning/blob/master/run1.mp4)
