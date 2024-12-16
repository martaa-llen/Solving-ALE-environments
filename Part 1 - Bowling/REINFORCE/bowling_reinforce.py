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

#init wandb and env
wandb.login(key="KEY")
gym.register_envs(ale_py)

#videos directory
if not os.path.exists("videos_bowling_reinforcement"):
    os.makedirs("videos_bowling_reinforcement")

#model directory
if not os.path.exists("models_reinforce"):
    os.makedirs("models_reinforce")


#init env
env = gym.make("ALE/Bowling-v5", render_mode="rgb_array")
print("Action space is {} ".format(env.action_space))
print("Observation space is {} ".format(env.observation_space))

#hyperparameters
INPUT_SHAPE = (1, 210, 160)
ACTION_DIM = env.action_space.n
LEARNING_RATE = 1e-4
GAMMA = 0.99
NUM_EPISODES = 10000
MAX_STEPS = 1000

#policy network
class PolicyNetwork(nn.Module):
    def __init__(self, input_shape, action_dim):
        super(PolicyNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.conv_output_size = self._get_conv_output(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(self.conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
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
    

REWARD_MULTIPLIERS = {
    'PIN_KNOCKED': 2.0,        
    'STRIKE': 10.0,            
    'CONSECUTIVE_STRIKE': 5.0, 
    'SPARE': 5.0,              
    'GUTTER_PENALTY': -0.5,    
    'SCORE_PROGRESS': 1.0      
}

class REINFORCEAgent:
    def __init__(self, input_shape, action_dim, learning_rate=3e-4, gamma=0.99):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.action_dim = action_dim  
        self.policy = PolicyNetwork(input_shape, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.entropy_coef = 0.01
        self.eps_start = 1.0
        self.eps_end = 0.01
        self.eps_decay = 500
        self.reward_multipliers = REWARD_MULTIPLIERS
        self.previous_pins_remaining = 10
        self.consecutive_strikes = 0
        self.frame_count = 0

    def preprocess_state(self, state):
        processed_state = np.mean(state, axis=2).astype(np.float32)
        processed_state = np.expand_dims(processed_state, axis=0)
        return torch.FloatTensor(processed_state).unsqueeze(0).to(self.device)
    def get_exploration_rate(self, episode):
        return self.eps_end + (self.eps_start - self.eps_end) * \
               math.exp(-1. * episode / self.eps_decay)
    
    def select_action(self, state, episode):
        #epsilon-greedy exploration
        if random.random() < self.get_exploration_rate(episode):
            action = random.randrange(self.action_dim)
            return action, None, None
        
        processed_state = self.preprocess_state(state)
        probs = self.policy(processed_state)
        probs = probs.clamp(min=1e-10, max=1.0)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item(), action_dist.log_prob(action), action_dist.entropy()
    
    def compute_returns(self, rewards):
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        return returns
    
    def update(self, log_probs, returns, entropies):
        if not log_probs:  #skip update if action was from exploration
            return 0.0
            
        policy_loss = []
        for log_prob, R, entropy in zip(log_probs, returns, entropies):
            policy_loss.append(-log_prob * R - self.entropy_coef * entropy)
        
        policy_loss = torch.stack(policy_loss).sum()
        
        #gradient clipping and scaling
        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        self.optimizer.step()
        
        return policy_loss.item()
    def shape_reward(self, reward, info, done, episode):
        """Modified reward shaping based on actual environment rewards"""
        shaped_reward = 0.0
        
        #base reward to prevent extremely negative starts
        shaped_reward = 0.1  #small positive baseline
        
        #og reward from the environment indicates pins knocked down
        pins_knocked = max(0, reward) 
        
        #base reward for any pins knocked down
        if pins_knocked > 0:
            shaped_reward += pins_knocked * self.reward_multipliers['PIN_KNOCKED']
            
            #strike (10 pins)
            if pins_knocked == 10:
                strike_reward = self.reward_multipliers['STRIKE']
                shaped_reward += strike_reward
                self.consecutive_strikes += 1
                consecutive_reward = (self.consecutive_strikes * 
                                    self.reward_multipliers['CONSECUTIVE_STRIKE'])
                shaped_reward += consecutive_reward
            
            #spare (remaining pins from previous throw)
            elif pins_knocked == self.previous_pins_remaining:
                spare_reward = self.reward_multipliers['SPARE']
                shaped_reward += spare_reward
        
        #store the remaining pins for spare detection
        self.previous_pins_remaining = 10 - pins_knocked if pins_knocked < 10 else 10
        
        #small reward for staying alive
        if not done:
            shaped_reward += 0.1
        
        #clip rewards to reasonable range
        shaped_reward = np.clip(shaped_reward, -1.0, 30.0)  
        
        return shaped_reward
    
    def reset_episode_stats(self):
        """Reset all episode-specific variables"""
        self.previous_pins_remaining = 10
        self.consecutive_strikes = 0
        self.frame_count = 0

if __name__ == "__main__":
   

    #init agent and wandb
    agent = REINFORCEAgent(INPUT_SHAPE, ACTION_DIM, LEARNING_RATE, GAMMA)
    wandb.init(
        project="bowling-reinforce",
        config={
            "algorithm": "REINFORCE",
            "num_episodes": NUM_EPISODES,
            "gamma": GAMMA
        }
    )

    #init metrics
    best_episode_reward = float('-inf')
    episode_rewards = []
    losses = []
    rolling_rewards = []
    rolling_losses = []
    best_episode_images = []
    #training loop
    total_rewards = []
    strikes = 0
    spares = 0

    for episode in range(NUM_EPISODES):
        if episode % 10 == 0:  #logs
            print(f"\n{'='*50}")
            print(f"Starting Episode {episode}")
            print(f"{'='*50}")
        
        should_record = (episode % 100 == 0) or (episode < 10)
        state = env.reset()[0]
        episode_reward = 0
        images = []
        step_rewards = []  #rewards for each step
        
        #reset agent stats
        agent.reset_episode_stats()
        
        for step in range(MAX_STEPS):
            #log every 100 steps
            if step % 100 == 0:
                print(f"\nEpisode {episode} - Step {step}")
                print(f"Current episode reward: {episode_reward:.2f}")
                print(f"Exploration Rate: {agent.get_exploration_rate(episode):.3f}")
                if step_rewards:  # If we have any rewards
                    print(f"Average step reward: {np.mean(step_rewards):.3f}")
                print(f"-"*30)
            
            #record frame
            if should_record:
                img = Image.fromarray(state)
                images.append(img)
            
            #sel action
            action, log_prob, entropy = agent.select_action(state, episode)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            shaped_reward = agent.shape_reward(reward, info, done, episode)
            step_rewards.append(shaped_reward)
            episode_reward += shaped_reward
            
            if done:
                break
            state = next_state
        
        #log ep
        if episode % 10 == 0:
            print(f"\n{'*'*50}")
            print(f"Episode {episode} Complete")
            print(f"Final Episode Reward: {episode_reward:.2f}")
            print(f"Best Episode Reward: {best_episode_reward:.2f}")
            print(f"Steps taken: {step+1}")
            if episode >= 99:
                rolling_avg = np.mean(episode_rewards[-100:])
                print(f"Rolling Average (100 ep): {rolling_avg:.2f}")
            print(f"Exploration Rate: {agent.get_exploration_rate(episode):.3f}")
            print(f"{'*'*50}\n")
            
            #log to wandb
            wandb.log({
                "episode": episode,
                "reward": episode_reward,
                "best_reward": best_episode_reward,
                "steps_taken": step + 1,
                "rolling_avg_reward": rolling_avg if episode >= 99 else 0,
                "exploration_rate": agent.get_exploration_rate(episode),
                "average_step_reward": np.mean(step_rewards)
            })
        
        #save best model and video
        if episode_reward > best_episode_reward:
            best_episode_reward = episode_reward
            print(f"\nNew Best Episode! Episode {episode}")
            print(f"New Best Reward: {best_episode_reward:.2f}")
            
            try:
                model_path = f"models/bestmodel_{episode}.pth"
                torch.save({
                    'episode': episode,
                    'model_state_dict': agent.policy.state_dict(),
                    'optimizer_state_dict': agent.optimizer.state_dict(),
                    'reward': best_episode_reward,
                }, model_path)
                
                if images:
                    best_video_path = f"videos_RL/best_episode_{episode}.gif"
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


    if not os.path.exists("videos_bowling_reinforcement"):
        os.makedirs("videos_bowling_reinforcement")
    if not os.path.exists("models_reinforce"):
        os.makedirs("models_reinforce")

    #save wandb
    wandb.save("videos_bowling_reinforcement/best_episode.gif")
    wandb.finish()
    env.close()

    #plot final results
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.plot(episode_rewards, alpha=0.3, label='Episode Reward')
    plt.plot(rolling_rewards, label='Rolling Average (100 episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(losses, alpha=0.3, label='Loss')
    plt.plot(rolling_losses, label='Rolling Average Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.hist(episode_rewards, bins=50, alpha=0.7)
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.title('Reward Distribution')

    plt.tight_layout()
    plt.show()

