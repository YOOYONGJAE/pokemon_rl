# eval_random.py
from pokemon_battle_env import PokemonBattleEnv
import numpy as np

def main():
    env = PokemonBattleEnv(render_mode=None)

    n_episodes = 10000

    total_rewards = []
    wins = 0
    losses = 0
    draws = 0

    print(">>>>> [BATTLE START] 완전 랜덤 배틀 시작")
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward

            done = terminated or truncated

        # 한 에피소드 끝났으니 승/패/무 판정
        if env.my_hp > 0 and env.opp_hp <= 0:
            wins += 1
        elif env.my_hp <= 0 and env.opp_hp > 0:
            losses += 1
        else:
            draws += 1

        total_rewards.append(ep_reward)

    avg_reward = sum(total_rewards) / len(total_rewards)

    print("\n>>> 랜덤 스킬 결투 결과")
    print(f"총 에피소드      : {n_episodes}")
    print(f"평균 보상    : {avg_reward:.3f}")
    print(f"승 / 패 / 무 : {wins} / {losses} / {draws}")
    print(f"승률      : {wins / n_episodes:.3f}")

    env.close()

if __name__ == "__main__":
    main()
