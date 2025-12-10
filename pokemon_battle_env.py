import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame_render
import battle_step

class PokemonBattleEnv(gym.Env):
    """
    reset() -> (obs, info)
    step()  -> (obs, reward, terminated, truncated, info)
    """
    metadata = {
        "render_modes": ["human"], # 휴먼모드, none cli 모드 존재
        "render_fps": 5,
    }
    
    def __init__(self, render_mode: str | None = None):
        super().__init__()
        self.render_mode = render_mode

        # 총 에너지랑 공격력, 방어력은 고정
        self.MAX_HP = 100
        self.BASE_ATK = 10
        self.BASE_DEF = 10

        # 기술 이름
        self.my_move_names = ["Tackle", "Leer", "Charm", "Tail Whip"]
        self.opp_move_names = ["Scratch", "Harden", "Milk Drink", "Explosion"]

        # 사용한 기술 TEMP
        self.last_my_move_name = ""
        self.last_opp_move_name = ""

        # 내 기술
        self.MY_MOVES = {
            0: {"name": "tackle", "base_acc": 0.80, "max_pp": 20},  # 몸통박치기
            1: {"name": "leer",    "base_acc": 1.00, "max_pp": 20},  # 노려보기
            2: {"name": "charm",    "base_acc": 1.00, "max_pp": 10},  # 애교부리기
            3: {"name": "tailwhip", "base_acc": 1.00, "max_pp": 10},  # 꼬리흔들기
        }

        # 상대 기술
        self.OPP_MOVES = {
            0: {"name": "scratch",   "base_acc": 0.90, "max_pp": 20},  # 할퀴기
            1: {"name": "harden",    "base_acc": 1.00, "max_pp": 10},  # 단단해지기
            2: {"name": "milk",      "base_acc": 1.00, "max_pp": 2},   # 우유마시기
            3: {"name": "explosion", "base_acc": 0.20, "max_pp": 5},   # 대폭발
        }

        # 방어력 / 버프 스택 상한 (정규화용)
        self.MAX_DEF = 40.0
        self.MAX_MY_ACC_STACK = 20.0
        self.MAX_OPP_ACC_DEBUFF = 10.0

        # 턴 제한
        self.max_turns = 100

        # ----- Gymnasium 공간 정의 -----
        # 상태: [my_hp, opp_hp, my_def, opp_def,
        #        my_pp(4), opp_pp(4),
        #        my_acc_buff_stack, opp_acc_debuff_stack]  → 총 14차원
        low = np.array([
            0.0, 0.0,                 # HP
            0.0, 0.0,                 # Def
            0.0, 0.0, 0.0, 0.0,       # my PP
            0.0, 0.0, 0.0, 0.0,       # opp PP
            0.0,                      # my_acc_buff_stack
            0.0                       # opp_acc_debuff_stack
        ], dtype=np.float32)

        high = np.array([
            1.0, 1.0,                 # HP
            1.0, 1.0,                 # Def
            1.0, 1.0, 1.0, 1.0,       # my PP
            1.0, 1.0, 1.0, 1.0,       # opp PP
            1.0,                      # my_acc_buff_stack
            1.0                       # opp_acc_debuff_stack
        ], dtype=np.float32)


        # [Gymnasium]에서 정의된 관측 공간
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # [Gymnasium]에서 정의된 행동 공간
        self.action_space = spaces.Discrete(4)  # 0~3: 네 기술

        # 난수 생성
        self.np_random, _ = gym.utils.seeding.np_random(None)

        # 내부 상태 변수들 (reset에서 초기화)
        self.my_hp = None
        self.opp_hp = None
        self.my_def = None
        self.opp_def = None
        self.my_pp = None
        self.opp_pp = None
        self.my_acc_buff_stack = None
        self.opp_acc_debuff_stack = None
        self.turn_count = None

        self.my_move_count = None

        ##### 파이게임 디자인 #####

        # self._init_battle_params()

        self.screen = None
        self.clock = None
        self.window_size = (640, 480)

        # 파이게임 출력 변수
        self.current_episode = 0
        self.total_episodes  = 0
        self.current_step    = 0
        self.total_timesteps_trained = 0         


    def _get_obs(self):
        my_hp_norm = self.my_hp / self.MAX_HP
        opp_hp_norm = self.opp_hp / self.MAX_HP
        my_def_norm = np.clip(self.my_def, 0, self.MAX_DEF) / self.MAX_DEF
        opp_def_norm = np.clip(self.opp_def, 0, self.MAX_DEF) / self.MAX_DEF

        my_pp_norm = np.array([
            self.my_pp[0] / self.MY_MOVES[0]["max_pp"],
            self.my_pp[1] / self.MY_MOVES[1]["max_pp"],
            self.my_pp[2] / self.MY_MOVES[2]["max_pp"],
            self.my_pp[3] / self.MY_MOVES[3]["max_pp"],
        ], dtype=np.float32)

        opp_pp_norm = np.array([
            self.opp_pp[0] / self.OPP_MOVES[0]["max_pp"],
            self.opp_pp[1] / self.OPP_MOVES[1]["max_pp"],
            self.opp_pp[2] / self.OPP_MOVES[2]["max_pp"],
            self.opp_pp[3] / self.OPP_MOVES[3]["max_pp"],
        ], dtype=np.float32)

        my_acc_norm = np.clip(self.my_acc_buff_stack, 0, self.MAX_MY_ACC_STACK) / self.MAX_MY_ACC_STACK
        opp_acc_norm = np.clip(self.opp_acc_debuff_stack, 0, self.MAX_OPP_ACC_DEBUFF) / self.MAX_OPP_ACC_DEBUFF

        obs = np.concatenate([
            np.array([my_hp_norm, opp_hp_norm, my_def_norm, opp_def_norm], dtype=np.float32),
            my_pp_norm,
            opp_pp_norm,
            np.array([my_acc_norm, opp_acc_norm], dtype=np.float32)
        ])

        return obs


    def _compute_damage(self, attacker_atk, skill_bonus, defender_def):
        base = (attacker_atk + skill_bonus) - defender_def * 0.5
        base = max(1.0, base)
        rand = self.np_random.uniform(0.85, 1.0) # 데미지 랜덤 조정
        return int(base * rand)


    def _sample_opp_action(self):
        if self.opp_hp <= 30 and self.opp_pp[2] > 0:
            return 2  # milk

        candidates = [] # 스킬후보군 (우유마시기 빼고)
        probs = []
        # 상대 행동의 확률 구분 (우유마시기는 상대 에너지에 기반하여 100% 발동하니까 제외했음)
        base_probs = {0: 0.7, 1: 0.2, 3: 0.1}

        for idx in [0, 1, 3]:
            if self.opp_pp[idx] > 0: # 사용가능한 스킬만
                candidates.append(idx)
                probs.append(base_probs[idx])

        if not candidates:
            return None

        probs = np.array(probs, dtype=np.float32)
        probs = probs / probs.sum()

        choice = self.np_random.choice(len(candidates), p=probs) # 확률 정해놨지만 랜덤성 추가
        return candidates[choice]


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

        self.last_my_move_name = ""
        self.last_opp_move_name = ""            

        self.my_hp = self.MAX_HP
        self.opp_hp = self.MAX_HP
        self.my_def = self.BASE_DEF
        self.opp_def = self.BASE_DEF

        self.my_pp = np.array([self.MY_MOVES[i]["max_pp"] for i in range(4)], dtype=np.int32)
        self.opp_pp = np.array([self.OPP_MOVES[i]["max_pp"] for i in range(4)], dtype=np.int32)

        self.my_acc_buff_stack = 0
        self.opp_acc_debuff_stack = 0

        self.turn_count = 0

        self.my_move_count = np.zeros(4, dtype=int)

        obs = self._get_obs()
        info = {}

        self.current_step = 0
        self.battle_ended = False

        return obs, info


    def step(self, action):
        return battle_step.step_core(self, action)


    # 파이게임을 통한 포켓몬 배틀 렌더링
    def _init_pygame(self): return pygame_render._init_pygame_core(self)
    def render(self): return pygame_render.render_core(self)
    def close(self): return pygame_render.close_core(self)
