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
import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordVideo
import datetime
from torch.distributions import Categorical
from collections import deque

#init wandb and env
wandb.login(key="KEY") 
gym.register_envs(ale_py)

#videos directory
if not os.path.exists("videos_RL"):
    os.makedirs("videos_RL")

#init env
env = gym.make("ALE/Boxing-v5", render_mode="rgb_array")
print("Action space is {} ".format(env.action_space))
print("Observation space is {} ".format(env.observation_space))


class BoxingPolicyNetwork(nn.Module):
    def __init__(self, input_shape, action_dim):
        super(BoxingPolicyNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv_output_size = self._get_conv_output(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(self.conv_output_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),  #dropout for regularization
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Softmax(dim=-1)
        )

    def _get_conv_output(self, shape):
        bs = 1
        input = torch.autograd.Variable(torch.rand(bs, *shape))
        output_feat = self.conv(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

#value network (critic) --> reduce variance
class BoxingValueNetwork(nn.Module):
    def __init__(self, input_shape):
        super(BoxingValueNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv_output_size = self._get_conv_output(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(self.conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  
        )

    def _get_conv_output(self, shape):
        bs = 1
        input = torch.autograd.Variable(torch.rand(bs, *shape))
        output_feat = self.conv(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

#include baseline in REINFORCE agent
class BoxingREINFORCEAgent:
    def __init__(self, input_shape, action_dim, learning_rate=1e-4, gamma=0.99):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.action_dim = action_dim
        self.policy = BoxingPolicyNetwork(input_shape, action_dim).to(self.device)
        self.value_net = BoxingValueNetwork(input_shape).to(self.device)
        
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=learning_rate)
        
        self.gamma = 0.995
        self.entropy_coef = 0.05
        self.value_coef = 0.5  #weight for value loss
        self.eps_start = 0.99
        self.eps_end = 0.1
        self.eps_decay = 5000
        
        #experience replay buffer
        self.memory = deque(maxlen=10000)  #store last 10000 transitions
        self.batch_size = 32
        self.min_replay_size = 1000  #min experiences before training

    def preprocess_state(self, state):
        #frame stacking 
        processed_state = np.mean(state, axis=2).astype(np.float32)
        processed_state = np.expand_dims(processed_state, axis=0)
        return torch.FloatTensor(processed_state).unsqueeze(0).to(self.device)

    def get_exploration_rate(self, episode):
        return self.eps_end + (self.eps_start - self.eps_end) * \
               math.exp(-1. * episode / self.eps_decay)
    
    def select_action(self, state, episode):
        if random.random() < self.get_exploration_rate(episode):
            action = random.randrange(self.action_dim)
            return action, None, None, None
        
        processed_state = self.preprocess_state(state)
        probs = self.policy(processed_state)
        value = self.value_net(processed_state)
        
        probs = probs.clamp(min=1e-10, max=1.0)
        action_dist = Categorical(probs)
        action = action_dist.sample()
        
        return (
            action.item(),
            action_dist.log_prob(action),
            action_dist.entropy(),
            value
        )

    def compute_returns(self, rewards, values):
        #reward shaping --> aggressive behavior
        shaped_rewards = []
        for r in rewards:
            #amplify positive rewards and reduce penalty of negative rewards
            shaped_r = r * 1.5 if r > 0 else r * 0.7
            shaped_rewards.append(shaped_r)
        
        returns = []
        advantages = []
        R = 0
        
        for r, v in zip(reversed(shaped_rewards), reversed(values)):
            R = r + self.gamma * R
            advantage = R - v.item()
            returns.insert(0, R)
            advantages.insert(0, advantage)
        
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        
        if len(returns) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
            
        return returns, advantages

    def store_transition(self, state, action, reward, next_state, done, log_prob=None, entropy=None, value=None):
        """Store a transition in the replay buffer"""
        self.memory.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'log_prob': log_prob,
            'entropy': entropy,
            'value': value
        })

    def sample_batch(self):
        """Sample a batch of transitions from replay buffer"""
        batch = random.sample(self.memory, min(self.batch_size, len(self.memory)))
        
        states = torch.stack([self.preprocess_state(t['state']).squeeze(0) for t in batch])
        next_states = torch.stack([self.preprocess_state(t['next_state']).squeeze(0) for t in batch])
        rewards = torch.FloatTensor([t['reward'] for t in batch]).to(self.device)
        dones = torch.FloatTensor([t['done'] for t in batch]).to(self.device)
        
        #filter out None values for policy gradient components
        policy_batch = [t for t in batch if t['log_prob'] is not None]
        if policy_batch:
            log_probs = torch.stack([t['log_prob'] for t in policy_batch])
            entropies = torch.stack([t['entropy'] for t in policy_batch])
            values = torch.stack([t['value'] for t in policy_batch])
        else:
            log_probs, entropies, values = None, None, None
            
        return states, next_states, rewards, dones, log_probs, entropies, values

    def update(self, log_probs, returns, advantages, values, entropies):
        if len(self.memory) < self.min_replay_size:
            return 0.0, 0.0
            
        #sample multiple batches for more stable training
        total_policy_loss = 0
        total_value_loss = 0
        n_updates = 4  
        
        for _ in range(n_updates):
            states, next_states, rewards, dones, batch_log_probs, batch_entropies, batch_values = self.sample_batch()
            
            if batch_log_probs is None:
                continue
                
            #returns and advantages for the batch
            batch_returns, batch_advantages = self.compute_returns(rewards.cpu().numpy(), batch_values)
            
            #policy loss
            policy_loss = []
            for log_prob, advantage, entropy in zip(batch_log_probs, batch_advantages, batch_entropies):
                policy_loss.append(-log_prob * advantage - self.entropy_coef * entropy)
            policy_loss = torch.stack(policy_loss).sum()
            
            #value loss
            value_targets = batch_returns.unsqueeze(-1)
            value_pred = torch.cat(batch_values)
            value_loss = nn.MSELoss()(value_pred, value_targets)
            
            #update networks
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            self.policy_optimizer.step()
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=1.0)
            self.value_optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            
        return total_policy_loss / n_updates, total_value_loss / n_updates

if __name__ == "__main__":
    #hyperparameters optimized for Boxing
    INPUT_SHAPE = (1, 210, 160)
    ACTION_DIM = env.action_space.n
    LEARNING_RATE = 1e-4 
    GAMMA = 0.995
    NUM_EPISODES = 10000  

    #init agent and wandb
    agent = BoxingREINFORCEAgent(INPUT_SHAPE, ACTION_DIM, LEARNING_RATE, GAMMA)
    wandb.init(
        project="boxing-reinforce2",
        config={
            "algorithm": "REINFORCE",
            "num_episodes": NUM_EPISODES,
            "gamma": GAMMA,
            "learning_rate": LEARNING_RATE
        }
    )

    #training
    episode_rewards = []
    best_mean_reward = float('-inf')

    #directories
    for directory in ["boxing_models_ev", "boxing_vids_ev"]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    #change simple env with a wrapper for video recording
    def make_env(video_folder, episode):
        env = gym.make("ALE/Boxing-v5", render_mode="rgb_array")
        if episode % 100 == 0:  
            env = RecordVideo(
                env, 
                video_folder=video_folder,
                episode_trigger=lambda x: True,  #record all episodes when wrapped
                name_prefix=f"episode_{episode}"
            )
        return env

    for episode in range(NUM_EPISODES):
        env = make_env("boxing_vids_ev", episode)
        state, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        log_probs = []
        rewards = []
        entropies = []
        values = []
        
        while not (done or truncated):
            #sel action
            action, log_prob, entropy, value = agent.select_action(state, episode)
            
            reward = 0
            for _ in range(4):
                next_state, r, done, truncated, _ = env.step(action)
                reward += r
                if done or truncated:
                    break
            
            if log_prob is not None:
                log_probs.append(log_prob)
                entropies.append(entropy)
                values.append(value)
            rewards.append(reward)
            episode_reward += reward
            state = next_state
        
        #update policy after episode
        if log_probs:
            returns, advantages = agent.compute_returns(rewards, values)
            policy_loss, value_loss = agent.update(log_probs, returns, advantages, values, entropies)
            
            episode_rewards.append(episode_reward)
            mean_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            
            #log metrics 
            wandb.log({
                "episode": episode,
                "reward": episode_reward,
                "mean_reward": mean_reward,
                "exploration_rate": agent.get_exploration_rate(episode),
                "policy_loss": policy_loss,
                "value_loss": value_loss
            })
        
        #save best model with timestamp
        if mean_reward > best_mean_reward and len(episode_rewards) >= 100:
            best_mean_reward = mean_reward

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(
                "boxing_models_ev", 
                f"best_boxing_model_{timestamp}_reward_{mean_reward:.2f}.pth"
            )
            torch.save(agent.policy.state_dict(), model_path)
            
            #record a video 
            eval_env = RecordVideo(
                gym.make("ALE/Boxing-v5", render_mode="rgb_array"),
                video_folder="boxing_vids_ev",
                name_prefix=f"best_model_{timestamp}_reward_{mean_reward:.2f}"
            )
            
            #record one episode with best model
            eval_state, _ = eval_env.reset()
            eval_done = False
            eval_truncated = False
            while not (eval_done or eval_truncated):
                eval_action, _, _, _ = agent.select_action(eval_state, episode)
                eval_state, _, eval_done, eval_truncated, _ = eval_env.step(eval_action)
            eval_env.close()
        
        #regular checkpoint every 1000 episodes
        if (episode + 1) % 1000 == 0:
            checkpoint_path = os.path.join(
                "boxing_models_ev", 
                f"checkpoint_episode_{episode+1}.pth"
            )
            torch.save({
                'episode': episode,
                'model_state_dict': agent.policy.state_dict(),
                'optimizer_state_dict': agent.policy_optimizer.state_dict(),
                'best_mean_reward': best_mean_reward,
                'episode_rewards': episode_rewards,
            }, checkpoint_path)
        
        env.close()
        
        #progress
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{NUM_EPISODES} | Reward: {episode_reward:.2f} | Mean Reward: {mean_reward:.2f}")

    env.close()
    wandb.finish()
