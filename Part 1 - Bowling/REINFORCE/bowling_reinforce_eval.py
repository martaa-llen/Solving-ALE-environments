#import packages
import gymnasium as gym
import torch
import numpy as np
from collections import deque
from bowling_reinforce import REINFORCEAgent, PolicyNetwork  #import from bowling_reinforce
import matplotlib.pyplot as plt

#hyperparameters
INPUT_SHAPE = (1, 210, 160)
ACTION_DIM = 6 

MODEL_PATH = "./Solving-ALE-environments/Part 1 - Bowling/REINFORCE/finetuned_bowling_models/best_model_63.pth" 

"""Other models from the models_reinforce/finetuned_bowling_models can be loaded and evaluated, 
                    but we chose our best performing one """
NUM_EVALUATIONS = 100  

#init env
env = gym.make("ALE/Bowling-v5", render_mode="rgb_array")

#load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = REINFORCEAgent(INPUT_SHAPE, ACTION_DIM)
agent.policy.load_state_dict(torch.load(MODEL_PATH, map_location=device)['model_state_dict'])
agent.policy.eval()  #eval mode

#mettrics init
total_pins_knocked = 0
total_strikes = 0
total_rewards = []
max_points = float('-inf')  
max_pins = float('-inf')    

for episode in range(NUM_EVALUATIONS):
    state = env.reset()[0]
    episode_reward = 0
    pins_knocked = 0
    strikes = 0

    while True:
        processed_state = agent.preprocess_state(state)
        action_probs = agent.policy(processed_state)
        action = action_probs.argmax().item()  #sel action highest probability
        next_state, reward, terminated, truncated, info = env.step(action)
        
        #current state
        plt.imshow(state)  
        plt.axis('off')  
        plt.title(f'Episode: {episode}, Reward: {episode_reward:.2f}')
        plt.pause(0.01)  

        #pins knocked down
        pins_knocked += max(0, reward)  #non-negative
        if pins_knocked == 10:  #strike
            strikes += 1
        
        episode_reward += reward
        state = next_state
        
        if terminated or truncated:
            break

    total_pins_knocked += pins_knocked
    total_strikes += strikes
    total_rewards.append(episode_reward)

    #update max points and max pins
    max_points = max(max_points, episode_reward)
    max_pins = max(max_pins, pins_knocked)

plt.close()

#averages
average_pins = total_pins_knocked / NUM_EVALUATIONS
average_reward = np.mean(total_rewards)

#eval results
print(f"Total Pins Knocked Down: {total_pins_knocked}")
print(f"Total Strikes Achieved: {total_strikes}")
print(f"Average Pins Knocked Down: {average_pins:.2f}")
print(f"Average Reward over {NUM_EVALUATIONS} episodes: {average_reward:.2f}")
print(f"Max Points in a Game: {max_points:.2f}")
print(f"Max Pins Knocked Down in a Game: {max_pins}")

env.close()