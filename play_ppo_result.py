# play_ppo.py
from stable_baselines3 import PPO
from pokemon_battle_env import PokemonBattleEnv

def main():
    env = PokemonBattleEnv(render_mode="")
    # env = PokemonBattleEnv(render_mode="human")

    model_path = "./models/ppo_pokemon_battle_best/best_model"
    model = PPO.load(model_path, env=env)

    n_episodes = 10000
    env.total_episodes = n_episodes

    total_rewards  = []
    wins = 0
    losses = 0
    draws = 0
    my_move_counts = {name: 0 for name in env.my_move_names}

    print(">>>>> [BATTLE START] PPO로 학습된 BEST 모델의 배틀 시작")

    for ep in range(n_episodes):
        env.current_episode = ep + 1
        obs, info = env.reset()
        done = False
        ep_reward  = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True) # 반환된 관측벡터(현상태), 가장 높은 확률로
            my_move_name = env.my_move_names[action]
            my_move_counts[my_move_name] += 1            

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward

            env.render()

            done = terminated or truncated

        total_rewards.append(ep_reward)

        if env.my_hp > 0 and env.opp_hp <= 0:
            wins += 1
        elif env.my_hp <= 0 and env.opp_hp > 0:
            losses += 1
        else:
            draws += 1 

        # 파이게임 데이터
        episodes_played = wins + losses + draws
        win_rate = wins / episodes_played if episodes_played > 0 else 0.0
        avg_reward_so_far = sum(total_rewards) / len(total_rewards)

        env.wins = wins
        env.losses = losses
        env.draws = draws
        env.win_rate = win_rate
        env.avg_reward = avg_reward_so_far


    # 로그 프린트
    avg_reward = sum(total_rewards) / len(total_rewards)
    print("\n>>> 베스트 모델을 이용한 결투 결과(PPO)")
    print(f"총 에피소드      : {n_episodes}")
    print(f"평균 보상    : {avg_reward:.3f}")
    print(f"승 / 패 / 무 : {wins} / {losses} / {draws}")
    print(f"승률      : {wins / n_episodes:.3f}")

    print("\n>>> 내 기술 사용 횟수 (PPO 에이전트)")
    total_my_moves = sum(my_move_counts.values())
    for name, cnt in my_move_counts.items():
        ratio = cnt / total_my_moves if total_my_moves > 0 else 0.0
        print(f"  {name:10s} : {cnt:7d}  ({ratio:6.2%})")

    env.close()

if __name__ == "__main__":
    main()
