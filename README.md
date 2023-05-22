# Nuturing-STEM-Robotics-and-RL
This repository includes the Python-based simulation program used in the robotics and reinforcement learning (RL) workshop designed for the University of Melbourne Capstone Project. It also includes scripts of different RL algorithms that were investigated, and their trianing and testing results.
# Program
The GUI is developed using [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter), a python UI-library based on Tkinter that provides a more modern theme. 
The GUI contains two tabs for the two OpenAI Gym environments that the agent could play. They are [Cart Pole](https://www.gymlibrary.dev/environments/classic_control/cart_pole/), and [Car Racing](https://www.gymlibrary.dev/environments/box2d/car_racing/). Each tab has the same layout with 3 frames. The left frame contains the selectable training parameters, and buttons to perform various functions. For Cart Pole, the seletables are learning rate and types of reward. For Car Racing, the selectable is just learning rate. The FPS parameter is not used in training, but used when the environments are played by keyboard. The middle frame contains the canvas that will plot the neural network model. The right frame contains a canvas that will plot agent's cumlative rewards over training episodes in real time, and it also contains a text box to print out verbose. The GUI contains 7 functions in total:
**Start Trianing:** Trains the agent with the chosen training parameters and envrionment over 100 episodes
Note: due to time constraints, actual training is not possible, so the function is just a replay of training session that was done beforehand
**Start Testing:** Test the agent in a new episode with the environment that it was trained in
**Reset to Default:** Reset the training parameters to default values, clear out all plots, and delete the active agent if there is any
**Load Network Model:** Load a pre-trained agent with the chosen training parameters and environment
**Log:** Openup a table window to visualize the log content of a training session
**Play:** Let user to play the chosen environment with the chosen FPS using keyboard. **SPACEBAR** to start, **ESC** to exit, **ARROW keys** to control.
**Replay:** Replay a past training epsiode specified by the entry box below

# Cartpole-v0 using DQN
![](https://github.com/JYL31/Nuturing-STEM-Robotics-and-RL/blob/main/Img/cartpole_eg.gif)

# Modified CarRacing-v2 using DQN
![](https://github.com/JYL31/Nuturing-STEM-Robotics-and-RL/blob/main/Img/car_racing_eg.gif)
