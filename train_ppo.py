# train_ppo.py
import os

import gymnasium as gym  # 안 써도 되지만 관례상 넣어둠
from pokemon_battle_env import PokemonBattleEnv

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback


def make_env():
    env = PokemonBattleEnv(render_mode=None)
    env = Monitor(env)
    return env

if __name__ == "__main__":
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)

    tmp_env = PokemonBattleEnv(render_mode=None)
    check_env(tmp_env, warn=True)
    tmp_env.close()

    env = make_env()

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,      
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,     
        verbose=1,
        tensorboard_log="./logs/ppo_pokemon_battle",
    )

    eval_env = make_env()
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/ppo_pokemon_battle_best",
        log_path="./logs/ppo_pokemon_battle_eval",
        eval_freq=10_000,      
        n_eval_episodes=20,    
        deterministic=True,
    )

    total_timesteps = 300_000
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    
    model.save("./models/ppo_pokemon_battle_final")

    env.close()
    eval_env.close()
