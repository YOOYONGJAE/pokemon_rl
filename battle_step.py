#battle_step.py
import numpy as np

def step_core(env, action):
    
        # 행동 유효성 검사. 
        # 행동에는 0~3의 정수가 와야 함 (4가지 기술 중 하나 선택).
        # 이 행동은 action_space에서 짐나지움에 정의함.
        assert env.action_space.contains(action), f"Invalid action {action}"

        env.current_step += 1

        env.turn_count += 1
        info = {}

        damage_to_opp = 0
        damage_to_me = 0

        env.last_my_move_name = env.my_move_names[action]

        # 내 스킬 턴
        if env.my_hp > 0:
            if env.my_pp[action] > 0:
                env.my_pp[action] -= 1

                env.my_move_count[action] += 1

                if action == 0:  # 몸통박치기 [설명 : 물리 공격, 기본 공격력 +10, 명중률 80% (버프에 따라 증가), 상대 HP 감소]
                    base_acc = env.MY_MOVES[0]["base_acc"]
                    current_acc = base_acc + 0.01 * env.my_acc_buff_stack
                    current_acc = float(np.clip(current_acc, 0.0, 0.99))
                    if env.np_random.random() < current_acc:
                        dmg = env._compute_damage(env.BASE_ATK, 10, env.opp_def)
                        env.opp_hp = max(0, env.opp_hp - dmg)
                        damage_to_opp += dmg

                elif action == 1:  # 노려보기 [설명 : 공격 계열 기술들의 명중률 +1 스택, 최대 스택까지 누적 (실제 명중률 = base_acc + 0.01*스택)]
                    env.my_acc_buff_stack = min(env.my_acc_buff_stack + 1, env.MAX_MY_ACC_STACK)

                elif action == 2:  # 애교부리기 [설명 : 상대 모든 기술의 명중률 -5% 디버프 1스택, 최대 스택까지 누적]
                    env.opp_acc_debuff_stack = min(env.opp_acc_debuff_stack + 1, env.MAX_OPP_ACC_DEBUFF)

                elif action == 3:  # 꼬리흔들기 [설명 : 상대 방어력 -5, 이후 물리 공격들이 더 큰 데미지]
                    env.opp_def -= 5

        # 상대 죽었으면 끝 (terminated)
        if env.opp_hp <= 0:
            terminated = True
            truncated = False
            reward = (damage_to_opp - damage_to_me) / 100.0 + 1.0
            obs = env._get_obs()
            return obs, reward, terminated, truncated, info

        # 상대턴
        if env.opp_hp > 0 and env.my_hp > 0:
            opp_action = env._sample_opp_action()
            # opp_action이 None일 수도 있으니 먼저 체크
            if opp_action is not None:
                env.last_opp_move_name = env.opp_move_names[opp_action]
            else:
                env.last_opp_move_name = "" 

            if opp_action is not None and env.opp_pp[opp_action] > 0:
                env.opp_pp[opp_action] -= 1
                if opp_action == 0:  # 할퀴기 [설명 : 물리 공격, 기본 공격력 +10, 명중률 90% (디버프에 따라 감소), 우리 HP 감소]
                    base_acc = env.OPP_MOVES[0]["base_acc"]
                    current_acc = base_acc - 0.05 * env.opp_acc_debuff_stack
                    current_acc = float(np.clip(current_acc, 0.1, 0.99))
                    if env.np_random.random() < current_acc:
                        dmg = env._compute_damage(env.BASE_ATK, 10, env.my_def)
                        env.my_hp = max(0, env.my_hp - dmg)
                        damage_to_me += dmg

                elif opp_action == 1:  # 단단해지기 [설명 : 상대(잠만보) 방어력 +5, 이후 들어오는 물리 피해 감소]
                    env.opp_def += 5

                elif opp_action == 2:  # 우유마시기 [설명 : 상대 HP 30 회복, 최대 HP(100)를 넘지 않음]
                    env.opp_hp = min(env.MAX_HP, env.opp_hp + 30)

                elif opp_action == 3:  # 대폭발 [설명 : 고위험 고화력 공격, 기본 공격력 +40, 낮은 명중률로 우리 HP에 큰 피해]
                    base_acc = env.OPP_MOVES[3]["base_acc"]
                    current_acc = base_acc - 0.05 * env.opp_acc_debuff_stack
                    current_acc = float(np.clip(current_acc, 0.05, 0.9))
                    if env.np_random.random() < current_acc:
                        dmg = env._compute_damage(env.BASE_ATK, 40, env.my_def)
                        env.my_hp = max(0, env.my_hp - dmg)
                        damage_to_me += dmg


        # 배틀 종료
        terminated = False
        truncated = False

        if env.my_hp <= 0 or env.opp_hp <= 0:
            terminated = True

        if env.turn_count >= env.max_turns:
            truncated = True

        reward = (damage_to_opp - damage_to_me) / 100.0

        if terminated or truncated:
            # 배틀 종료 리워드
            env.battle_ended = True
            if env.my_hp > 0 and env.opp_hp <= 0:
                reward += 1.0
            elif env.my_hp <= 0 and env.opp_hp > 0:
                reward -= 1.0
            else:
                reward -= 0.2

        obs = env._get_obs()
        return obs, reward, terminated, truncated, info