# Taxi-v3
 Hello guys! Welcome to my practice projects. I am interested in AI development, so I will try to implement algorithms and models in the field of AI. I will share my experience and results with you. Feel free to check codes and feedback. If you have any comments or suggestions, please let me know. Thank you!

## Project Description
 I need to implement deep-qlearning in the taxi environment. The taxi will pick up and drop off passengers at random locations. There are 4 locations (labeled by different letters) and your job is to pick up the passenger at one location and drop them off in another. You receive +20 points for a successful dropoff, and lose 1 point for every timestep it takes. There is also a 10 point penalty for illegal pick-up and drop-off actions. You need to implement an agent that learns to pick up passengers at one location and drop them off in another.

## Installation
1. Clone the repository
2. Install the requirements

## Environment
* Python 3.9.17
* gymnasium 1.1.1

## Training Feedback
1. An interesting phenomenon was observed during the training process. The model doesn't do any pickup and dropoff operation in order to prevent high negative rewards. Therefore, in the whole training process, the model seems like to learn a passive strategy about moving until the maximum steps of a episode.
2. Observing the loss during the training, we can see the performance of the model is going to converge, the loss decreasing from 678 to 0.6. However, based on the reward from -700 to -200, the reward doesn't keep increasing, and it's stable around -200 to -300. Due the decreasing of the loss, it doesn't seem like underfitting. Therefore, we need to consider reshaping the reward setting. Try to increase the reward in the following training.
3. After doubling the reward of a episode, the similar phenomenon is observed. However, there are some changes observed in the episode lengths graph. There are some episodes finished within or near 100 steps, which means that the model probably needs more exploration to learn. Due to the reward only given at the end of an episode, the model doesn't learn about the environment very efficiently. Therefore, I probably need to update the reward setting again. In my opinion, the reward could be also given when the taxi picks up a passenger successfully. This could help the model learn the goal of the environment, and accelerate the learning process.