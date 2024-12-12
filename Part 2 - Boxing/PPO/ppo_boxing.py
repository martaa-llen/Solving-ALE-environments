#import packages 
import gymnasium as gym
import ale_py

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv, NoopResetEnv
import wandb
from wandb.integration.sb3 import WandbCallback
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import deque


gym.register_envs(ale_py) #to be able to use ale environments we have to register them in gym

#create environment
env = gym.make("ALE/Boxing-v5", render_mode="rgb_array")  

#to save metrics in wandb
wandb.login(key="067eada2bb47e4ae47f13cdb62ae8ab49d182618")

#hyperparameters  used
config = {
    "algorithm": "PPO",
    "env_id": "ALE/Boxing-v5",
    "total_timesteps": 2_000_000,  
    "policy_type": "CnnPolicy",
    "n_envs": 16,  #8 --> 16 (faster parallel collection)
    "n_steps": 128,
    "batch_size": 512,  #bigger batch size
    "n_epochs": 4,
    "learning_rate": 2.5e-4,  
    "clip_range": 0.2,  #standard PPO clip range
    "vf_coef": 0.5,
    "ent_coef": 0.01,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "normalize_advantage": True,
    "max_grad_norm": 0.5,
    "frame_stack": 4
}

#initialize wandb with config params
run = wandb.init(
    project="boxing-sb3",
    config=config,
    sync_tensorboard=True,
    monitor_gym=True,
    save_code=True
)

def make_env(env_id, rank, seed=0):
    def _init():
        env = gym.make(env_id, render_mode="rgb_array")
        env = NoopResetEnv(env, noop_max=30)  #randomness to initial state
        env = MaxAndSkipEnv(env, skip=4)  #frame skipping
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init

#create directories needed for saving models, logs and videos 
def setup_directories():
    directories = [
        "./models_boxing_ppo4",
        "./models_boxing_ppo4/best_model_boxing_ppo4",
        "./models_boxing_ppo4/checkpoints_boxing_ppo4",
        "./videos_boxing_ppo4",
        "./logs_boxing_ppo4"
    ]
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)

class EpisodeRecorder:
    def __init__(self, save_dir="videos_boxing_ppo4/"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def save_video(self, images, episode, total_reward=None, is_best=False):
        if not images:
            return
        
        if is_best: 
            path = f"{self.save_dir}best_episode_{episode}_reward_{total_reward:.2f}.gif" #path if we obtain better rewards, save video with specific name 
        else:
            path = f"{self.save_dir}episode_{episode}.gif" #path save regular episodes 
        
        images[0].save(path, save_all=True, append_images=images[1:], duration=100, loop=0)
        wandb.log({f"Episode {episode} Video": wandb.Video(path, caption=f"Episode {episode} - Reward: {total_reward:.2f}")})
        return path #path of the video saved

class MetricLogger:
    def __init__(self, window_size=100):
        self.rewards = []
        self.actor_losses = []
        self.critic_losses = []
        self.entropy_losses = []
        self.rolling_rewards = deque(maxlen=window_size)
        self.avg_rewards = []
        
    def log_step(self, reward, actor_loss=None, critic_loss=None, entropy_loss=None):
        self.rewards.append(reward)
        self.rolling_rewards.append(reward)
        self.avg_rewards.append(sum(self.rolling_rewards) / len(self.rolling_rewards))
        
        if actor_loss is not None:
            self.actor_losses.append(actor_loss)
        if critic_loss is not None:
            self.critic_losses.append(critic_loss)
        if entropy_loss is not None:
            self.entropy_losses.append(entropy_loss)
    
    def plot_metrics(self, save_dir="./plots_boxing_ppo4"):
        os.makedirs(save_dir, exist_ok=True)
        
        #plot rewards
        plt.figure(figsize=(10, 5))
        plt.plot(self.rewards, label='Episode Reward', alpha=0.6)
        plt.plot(self.avg_rewards, label='Rolling Average Reward', linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Training Rewards')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{save_dir}/rewards.png")
        plt.close()
        

def train():
    setup_directories()
    logger = MetricLogger(window_size=100)  #create logger
    
    #create environments
    env = DummyVecEnv([make_env(config["env_id"], i) for i in range(config["n_envs"])])
    env = VecFrameStack(env, n_stack=config["frame_stack"])
    env = VecTransposeImage(env)
    
    eval_env = DummyVecEnv([make_env(config["env_id"], 0)])
    eval_env = VecFrameStack(eval_env, n_stack=config["frame_stack"])
    eval_env = VecTransposeImage(eval_env)
    
    
    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=config["learning_rate"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_range=config["clip_range"],
        clip_range_vf=config["clip_range"],
        normalize_advantage=config["normalize_advantage"],
        ent_coef=config["ent_coef"],
        vf_coef=config["vf_coef"],
        max_grad_norm=config["max_grad_norm"],
        verbose=1,
        tensorboard_log=f"runs_boxing_ppo4/{wandb.run.id}"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models_boxing_ppo4/best_model_boxing_ppo4",
        log_path="./logs_boxing_ppo4",
        eval_freq=1000,
        deterministic=True,
        render=False,
        n_eval_episodes=5,
        verbose=1
    )

    class LoggerCallback(WandbCallback):
        def _on_step(self) -> bool:
            if self.model.num_timesteps % 1000 == 0:
                #latest episode reward
                if len(self.model.ep_info_buffer) > 0:
                    reward = self.model.ep_info_buffer[-1]["r"]
                    logger.log_step(
                        reward=reward,
                        actor_loss=self.model.logger.name_to_value.get("train/actor_loss"),
                        critic_loss=self.model.logger.name_to_value.get("train/critic_loss"),
                        entropy_loss=self.model.logger.name_to_value.get("train/entropy_loss")
                    )
            return True

    callbacks = [eval_callback, LoggerCallback()]

    try:
        model.learn(
            total_timesteps=config["total_timesteps"],
            callback=callbacks,
            progress_bar=True
        )
        
        #save plots training
        logger.plot_metrics()
        print("Training plots saved in ./plots_boxing_ppo4/")
        
        final_model_path = f"./models_boxing_ppo4/boxing_final_model_ppo4_{wandb.run.id}"
        model.save(final_model_path)
        print(f"Final model saved to {final_model_path}")

        wandb.save(final_model_path)

        best_model_path = "./models_boxing_ppo4/best_model_boxing_ppo4/best_model.zip"
        if os.path.exists(best_model_path):
            wandb.save(best_model_path)
            print(f"Best model logged to wandb: {best_model_path}")

    except Exception as e:
        print(f"Training interrupted: {e}")
        #try to save plots if training interrupted
        logger.plot_metrics()

    finally:
        env.close()
        eval_env.close()


if __name__ == "__main__":
    
    setup_directories()  #necessary directories
    
    train()
    
