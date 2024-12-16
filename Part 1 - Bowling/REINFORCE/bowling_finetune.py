#import packages
import gymnasium as gym
import numpy as np
import torch
import os
from PIL import Image
import wandb
import ale_py  
from bowling_reinforce import REINFORCEAgent, INPUT_SHAPE, ACTION_DIM, LEARNING_RATE, GAMMA  #import from bowling_reinforce

#init wandb
wandb.init(
    project="finetuned_bowling",
    config={
        "algorithm": "REINFORCE",
        "num_episodes": 1000,
        "gamma": GAMMA
    }
)

#register ALE environments
gym.register_envs(ale_py)  

#create directories for saving models and videos
if not os.path.exists("finetuned_bowling_models"):
    os.makedirs("finetuned_bowling_models")
if not os.path.exists("finetuned_bowling_videos"):
    os.makedirs("finetuned_bowling_videos")

#load best model
best_model_path = "./Solving-ALE-environments/Part 1 - Bowling/REINFORCE/models_reinforce/bestmodel_8411.pth"
agent = REINFORCEAgent(INPUT_SHAPE, ACTION_DIM, LEARNING_RATE, GAMMA)

if os.path.exists(best_model_path):
    #map_location to load the model on CPU
    checkpoint = torch.load(best_model_path, map_location=torch.device('cpu'))
    agent.policy.load_state_dict(checkpoint['model_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("Loaded best model from", best_model_path)

#init env
env = gym.make("ALE/Bowling-v5", render_mode="rgb_array")

#continue training agent
NUM_EPISODES = 1000  #num eps to continue training
best_episode_reward = float('-inf')

for episode in range(NUM_EPISODES):
    state = env.reset()[0]
    episode_reward = 0
    agent.reset_episode_stats()
    images = []  #store frames for video

    for step in range(1000):  
        action, log_prob, entropy = agent.select_action(state, episode)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        shaped_reward = agent.shape_reward(reward, info, done, episode)
        episode_reward += shaped_reward
        
        #record frame for video
        img = Image.fromarray(state)
        images.append(img)

        if done:
            break
        state = next_state
    
    #save best model if current episode reward is better
    if episode_reward > best_episode_reward:
        best_episode_reward = episode_reward
        print(f"\nNew Best Episode! Episode {episode}")
        print(f"New Best Reward: {best_episode_reward:.2f}")
        
        try:
            model_path = f"finetuned_bowling_models/best_model_{episode}.pth"
            torch.save({
                'episode': episode,
                'model_state_dict': agent.policy.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'reward': best_episode_reward,
            }, model_path)

            #save video of the best episode
            best_video_path = f"finetuned_bowling_videos/best_vid_{episode}.gif"
            images[0].save(
                best_video_path,
                save_all=True,
                append_images=images[1:],
                duration=100,
                loop=0
            )
            wandb.log({
                "best_episode_video": wandb.Video(best_video_path, fps=10, format="gif"),
                "best_episode": episode,
                "best_reward": best_episode_reward
            })
        except Exception as e:
            print(f"Error saving model or video: {e}")

env.close()
wandb.finish()
