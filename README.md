# Nuturing-STEM-Robotics-and-RL
This repository includes the Python-based simulation program used in the robotics and reinforcement learning (RL) workshop designed for the University of Melbourne Capstone Project. It also includes scripts of different RL algorithms that were investigated, and their trianing and testing results.
# Program
![](https://github.com/JYL31/Nuturing-STEM-Robotics-and-RL/blob/main/Img/GUI.JPG "GUI")

The GUI is developed using [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter), a python UI-library based on Tkinter that provides a more modern theme. 

The GUI contains two tabs for the two OpenAI Gym environments that the agent could play. They are [Cart Pole](https://www.gymlibrary.dev/environments/classic_control/cart_pole/), and [Car Racing](https://www.gymlibrary.dev/environments/box2d/car_racing/). The GUI contains 7 functions in total:

**Start Trianing:** Trains the agent with the chosen training parameters and envrionment over 100 episodes. **Note:** due to time constraints, actual training is not possible, so the function is just a replay of training session that was done beforehand

**Start Testing:** Test the agent in a new episode with the environment that it was trained in

**Reset to Default:** Reset the training parameters to default values, clear out all plots, and delete the active agent if there is any

**Load Network Model:** Load a pre-trained agent with the chosen training parameters and environment

**Log:** Openup a table window to visualize the log content of a training session

**Play:** Let user to play the chosen environment with the chosen FPS using keyboard. **SPACEBAR** to start, **ESC** to exit, **ARROW keys** to control.

**Replay:** Replay a past training epsiode specified by the entry box below

## To Run the Program
1. Clone the workshop branch
```bash
git clone -b workshop https://github.com/JYL31/Nuturing-STEM-Robotics-and-RL.git
```
2. Use `Conda` to activate the python environment for the program
```bash
conda env create -f STEM_workshop.yml
conda activate STEM_workshop
```
3. Run the Python script
```bash
python main.py
```

# Cartpole-v0 using DQN
![](https://github.com/JYL31/Nuturing-STEM-Robotics-and-RL/blob/main/Img/cartpole_eg.gif "agent play cartpole")
![](https://github.com/JYL31/Nuturing-STEM-Robotics-and-RL/blob/main/Code/Workshop/Cartpole/model_plot.png "network model for cartpole")
Simple fully connected neural network model.
* **Input:** cart position, cart velocity, pole angle, pole angular velocity
* **Output:** action 0: left, 1: right

# CarRacing-v2 using DQN
![](https://github.com/JYL31/Nuturing-STEM-Robotics-and-RL/blob/main/Img/car_racing_eg.gif "agent play car racing")
![](https://github.com/JYL31/Nuturing-STEM-Robotics-and-RL/blob/main/Code/Workshop/CartRacing/model_plot.png "network model for car racing")
Standard convolution neural network model (CNN).
* **Input:** a stack of 96-by-96 pixels image in 5 consecutive frames
* **Output:** action (steering, gas, break): \[(0, 1, 0), (-1, 0, 0), (0, 0, 0), (1, 0, 0), (0, 0, 0.5)\]
