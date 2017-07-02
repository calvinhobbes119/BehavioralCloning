## Project: Behavioral Cloning Project [![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Model Architecture
---
I used Nvidia's end-to-end Self-Driving Deep Learning network for the Behavioral Cloning project. The network has 5 convolutional layers and 3 fully connected (dense) layers. To prevent overfitting, as suggested by the project rubric, I inserted dropouts between the first two dense (fully-connected) layers for regularization with dropout probability of 0.3. Determining a good network architecture to solve a given problem appears to be more of an art than a science, and involves extensive experimentation. For this project, I relied on existing literature to find a good starting point for my network architecture. So I chose to start with the Nvidia architecutre, which has shown very promising results. It turned out that this architecture was sufficient for successfully achieving the objectives of this project, so I did not have to evaluate variations of the architecture. However, I did explore changing the dropout probability to 0.5 (the performance was worse with this on that dataset I had) before settling on a value of 0.3 which appeared to work better. It is possible that with a more complex problem (like on the challenge track) requiring more extensive training data, a dropout probability of 0.5 may work better to avoid overfitting, but this is still under investigating.

![network](https://github.com/calvinhobbes119/BehavioralCloning/blob/master/DriveNetwork.png) 

Training, Validation and Testing
---
I started out by splitting the dataset (which was included with the Udacity project) using an 80/20 ratio into training and validation sets. I used the center image as well as the left and right camera images from this dataset with a correction of -/+ 0.1 respectively. I initially tried a correction value of -/+ 0.2, but that did not work well. In order to augment my dataset, I also flipped these three images horizontally (using the negative of the steering angle as the corresponding training output). I used the Adam optimizer with a batch size of 32 samples, and training for 7 epochs. After observing the performance of the network I collected additional data on stretches of the track where the car was veering off course, and included them in my dataset. I repeated this procedure until the car was able to go completely around the track without going off course. My final training dataset was ~66K image samples, and my validation set was ~14K image samples.

[![Augmented Data Set 1](https://github.com/calvinhobbes119/BehavioralCloning/blob/master/Untitled.png)](https://youtu.be/RFD8soBKVxM)

Future improvements
---
I plan to experiment using more data from the challenge track so the car successfully completes the challenge course as well. Currently the car makes it way through roughly 10% of the challenge course before veering off track. The training and validation losses are still fairly high after 20 epochs of training, indicating that the network weights have not yet converged. I am currently testing by increasing the number of epochs as well as collecting more data on the challenge track.
