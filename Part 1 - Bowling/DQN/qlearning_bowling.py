#import packages
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
from collections import namedtuple
import wandb
import os
from PIL import Image
import ale_py

#to save metrics in wandb
wandb.login(key="KEY")

gym.register_envs(ale_py) #to be able to use ale environments we have to register them in gym

env = gym.make("ALE/Bowling-v5", render_mode="rgb_array")

print("Action space is {} ".format(env.action_space))
print("Observation space is {} ".format(env.observation_space))

#to save videos of performance during training
if not os.path.exists("videos_dqn"):
    os.makedirs("videos_dqn")
    
#DQN model
class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        
        #conv layers 
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        
        #output size of conv layers 
        conv_out_size = self._get_conv_out(input_shape)

        #sully connected layers 
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
    
    #to calculate output size of conv layers 
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)

#to store experiences
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

#replay buffer class 
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity #max experience that can save
        self.position = 0
        
    def push(self, experience):
        #to add experiences 
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size) #random experiences 
    
    def __len__(self):
        return len(self.buffer)


#hyperparameters  used
config = {
    "algorithm": "DQN",
    "BATCH_SIZE": 64,
    "GAMMA": 0.99,
    "EPSILON_START": 1.0,
    "EPSILON_END": 0.01,
    "EPSILON_DECAY": 0.9995,
    "TARGET_UPDATE": 500,  
    "REPLAY_BUFFER_SIZE": 5000,  
    "LEARNING_RATE": 1e-4,
    "NUM_EPISODES": 15000,
    "INPUT_SHAPE": (1, 84, 84),  #smaller input shape
    "MAX_STEPS": 1000
}

#initialize wandb with config params 
wandb.init(
    project="bowling-ale",
    config=config
)

#preprocess obs 
def preprocess_observation(observation):
    #convert obs to grayscale and resize to smaller shape 
    processed_obs = np.mean(observation, axis=2).astype(np.float32) #grayscale 
    processed_obs = torch.FloatTensor(processed_obs)
    return torch.nn.functional.interpolate(processed_obs.unsqueeze(0).unsqueeze(0), 
                                         size=(84, 84), 
                                         mode='bilinear', 
                                         align_corners=False).squeeze(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #to use gpu

n_actions = env.action_space.n #num possible actions in env

#init networks
policy_net = DQN(config["INPUT_SHAPE"], n_actions).to(device) #primary network
target_net = DQN(config["INPUT_SHAPE"], n_actions).to(device) #target network 
target_net.load_state_dict(policy_net.state_dict()) #use primary network weights in target network

#init optimizer and replay buffer
optimizer = optim.Adam(policy_net.parameters(), lr=config["LEARNING_RATE"])
memory = ReplayBuffer(config["REPLAY_BUFFER_SIZE"])

#function to select next actions (epsilon-greedy strategy)
def select_action(state, epsilon):
    if random.random() > epsilon:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1) #sel action with highest value
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device) #sel random action


def optimize_model():
    if len(memory) < config["BATCH_SIZE"]:
        return None #not enough experiences --> nothing to do 
    
    #sample experiences
    transitions = memory.sample(config["BATCH_SIZE"])
    batch = Experience(*zip(*transitions))
    
    #create mask (we want to know if the next state is non-terminal)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]) #put all non-terminal together 
    
    #convert to tensors 
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    
    #q-values using primary network (of action taken)
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    
    #max q-values using target network (next states)
    next_state_values = torch.zeros(config["BATCH_SIZE"], device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    #expected q-values
    expected_state_action_values = (next_state_values * config["GAMMA"]) + reward_batch
    
    #loss between q-values predicted vs expected
    loss = nn.SmoothL1Loss()(state_action_values, expected_state_action_values.unsqueeze(1))
    
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    
    return loss.item()
    

def step_environment(env, action, skip_frames=4): 
    total_reward = 0
    for _ in range(skip_frames): #do action fro specific num frames
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            return observation, total_reward, terminated, truncated, info
    return observation, total_reward, terminated, truncated, info

class EpisodeRecorder:
    def __init__(self, save_dir="videos_dqn/"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def should_record(self, episode, num_episodes):
        return (episode % 100 == 0 or  #should record episode each time it passes 100 
                episode == num_episodes - 1)  #should record Last episode
                
    def save_video(self, images, episode, total_reward=None, is_best=False):
        if not images:
            return
            
        if is_best:
            path = f"{self.save_dir}best_episode_{episode}.gif" #path if we obtain better rewards, save video with specific name 
        else:
            path = f"{self.save_dir}episode_{episode}.gif" #path save regular episodes 
            
        images[0].save(path, save_all=True, append_images=images[1:], duration=100, loop=0) #save video
        return path #path of the video saved

def train():
    recorder = EpisodeRecorder() #start recording class
    rewards_history = []
    best_reward = float('-inf')
    epsilon = config["EPSILON_START"]
    steps_done = 0

    for episode in range(config["NUM_EPISODES"]):
        state = env.reset()[0]
        state = preprocess_observation(state).unsqueeze(0).to(device)
        total_reward = 0
        episode_loss = 0
        images = [] #to collect img from the whole episode
        num_loss_updates = 0

        for step in range(config["MAX_STEPS"]):
            #sel and execut action
            action = select_action(state, epsilon)
            observation, reward, terminated, truncated, _ = step_environment(env, action.item())
            done = terminated or truncated
            total_reward += reward

            images.append(Image.fromarray(observation, 'RGB')) #append frames 

            #next state
            reward = torch.tensor([reward], device=device)
            next_state = None if done else preprocess_observation(observation).unsqueeze(0).to(device)
            
            memory.push(Experience(state, action, reward, next_state, done))
            state = next_state
            
            loss = optimize_model()
            if loss is not None:
                episode_loss += loss
                num_loss_updates += 1
            
            if done:
                break
                
            steps_done += 1
            
            #update target network 
            if steps_done % config["TARGET_UPDATE"] == 0:
                target_net.load_state_dict(policy_net.state_dict())
        
        #calculate metrics
        avg_loss = episode_loss / num_loss_updates if num_loss_updates > 0 else 0
  
        epsilon = max(config["EPSILON_END"], epsilon * config["EPSILON_DECAY"]) #epsilon update 
            
        #update rewards list
        rewards_history.append(total_reward)
        
        #log metrics
        metrics = {
            "episode": episode,
            "reward": total_reward,
            "epsilon": epsilon,
            "average_loss": avg_loss
        }
        
        if len(rewards_history) >= 100:
            metrics["rolling_avg_reward"] = sum(rewards_history[-100:]) / 100 #rolling average rewards (100 episodes)
            
        wandb.log(metrics) #update values in wandb 
        
        #save videos
        if images:
            if total_reward > best_reward:
                best_reward = total_reward
                #when we obtain a reward higher than the best one saved, we save the video of its performance
                path = recorder.save_video(images, episode, is_best=True) 
                print(f"New best reward: {best_reward}! Saved to {path}")
                
                save_checkpoint(policy_net, optimizer, episode, total_reward, is_best=True) #checkpoint of the best model saved 

                wandb.log({"best_video": wandb.Video(path)}) #we also save it on wandb

            #if it is the last episode or it is % 100 it should be recordered
            elif recorder.should_record(episode, config["NUM_EPISODES"]):
                path = recorder.save_video(images, episode)
                print(f"Saved episode {episode} video to {path}")

                wandb.log({"episode_video": wandb.Video(path)}) 

        #to save regular checkpoint (every 100 episodes)
        if episode % 100 == 0:  
            save_checkpoint(policy_net, optimizer, episode, total_reward, is_best=False)
            if len(rewards_history) >= 100:
                rolling_reward = sum(rewards_history[-100:]) / 100
                print(f"Episode {episode} - loss: {avg_loss}, Reward: {total_reward}, rolling avg reward: {rolling_reward}") 
            else:
                print(f"Episode {episode} - loss: {avg_loss}, Reward: {total_reward}")

        if episode == config["NUM_EPISODES"] - 1:
            last_episode_path = f"videos_dqn/final_episode.gif" #last episode saved with specific name 
            if images:
                images[0].save(last_episode_path, save_all=True, append_images=images[1:], duration=100, loop=0)
                print(f"Saved final episode to {last_episode_path}")
            
            #save final model state
            save_checkpoint(policy_net, optimizer, episode, total_reward, is_best=False)
            print("Saved final model checkpoint")

def save_checkpoint(model, optimizer, episode, reward, is_best=False):
    checkpoint = {
        'episode': episode,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'reward': reward,
    }

    #create models directory if it doesn't exist
    if not os.path.exists('models_dqn'):
        os.makedirs('models_dqn')
    
    #save latest model locally and to wandb
    local_path = os.path.join('models_dqn', 'latest_model_dqn.pth')
    wandb_path = os.path.join(wandb.run.dir, 'latest_model_dqn.pth')
    torch.save(checkpoint, local_path)
    torch.save(checkpoint, wandb_path)
    
    if is_best:
        #save best model locally and to wandb
        local_best_path = os.path.join('models_dqn', 'best_model_dqn.pth')
        wandb_best_path = os.path.join(wandb.run.dir, 'best_model_dqn.pth')
        torch.save(checkpoint, local_best_path)
        torch.save(checkpoint, wandb_best_path)
        wandb.save('best_model_dqn.pth')
        print(f"Saved best model to {local_best_path}")

train()

env.close()
wandb.finish()