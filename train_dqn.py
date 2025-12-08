# train_dqn.py
import os

import gymnasium as gym
from pokemon_battle_env import PokemonBattleEnv

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback


def make_env():
    env = PokemonBattleEnv(render_mode=None)
    # 래퍼 = 환경을 감싸서 기록을 추가하는 장치
    env = Monitor(env)
    return env

if __name__ == "__main__":
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)

    tmp_env = make_env()
    check_env(tmp_env, warn=True)  
    tmp_env.close()

    env = make_env()

    model = DQN(
        "MlpPolicy",          
        env,
        learning_rate=1e-3,   
        buffer_size=100_000,  
        batch_size=64,
        gamma=0.99,
        train_freq=4,         
        gradient_steps=1,
        target_update_interval=1_000,
        exploration_fraction=0.3,   
        exploration_final_eps=0.05, 
        verbose=1,
        tensorboard_log="./logs/dqn_pokemon_battle",  
    )

    eval_callback = EvalCallback(
        env,
        best_model_save_path="./models/dqn_pokemon_battle_best",
        log_path="./logs/dqn_pokemon_battle_eval",
        eval_freq=10_000,
        n_eval_episodes=20,
        deterministic=True,
    )

    model.learn(total_timesteps=300_000, callback=eval_callback)

    total_timesteps = 300_000  
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    model.save("./models/dqn_pokemon_battle_final")

    env.close()
