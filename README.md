
# Pokemon Reinforcement Learning Battle

단순화된 포켓몬 1:1 배틀 환경(`PokemonBattleEnv`)에서  
강화학습 에이전트(DQN, PPO)를 학습·평가하는 프로젝트입니다.

- 환경은 **Gymnasium 호환** 클래스
- 알고리즘은 **Stable-Baselines3 (DQN, PPO)** 
- 학습된 정책은 **파이게임(Pygame)** 으로 시각화하여 실제 배틀처럼 확인할 수 있습니다.

---

## 1. 실행 환경 & 설치

### Python 버전
- 파이env를 사용하여 파이썬 3.12.3 환경에서 수행했습니다.

### 의존성 설치

```bash
pip install -r requirements.txt
````

`requirements.txt`에는 대략 다음 패키지가 포함되어 있습니다.

* `stable-baselines3`
* `gymnasium`
* `numpy`
* `pygame`
* `torch` (CPU만으로도 실행 가능)
* `tensorboard` (학습 곡선 시각화용)

> **참고:** CUDA(GPU)를 쓰지 않는 환경이라면, PyTorch 설치 시 CPU 전용 버전을 사용해도 충분
> 설치하면서 같이 들어왔지만 CUDA를 쓰진 않았습니다.

### 에셋(이미지) 준비

파이게임 렌더링에서 `assets/` 폴더의 이미지를 사용합니다.

* `assets/Bulbasaur.png`
* `assets/Snorlax.png`

이미지 파일 경로나 파일명을 바꾸고 싶다면 `pygame_render.py` 내부의 로드 경로를 함께 수정해 주세요.

---

## 2. 환경 개요 (`PokemonBattleEnv`)

* 파일: `pokemon_battle_env.py`
* Gymnasium 스타일 인터페이스:

  * `reset() -> (obs, info)`
  * `step(action) -> (obs, reward, terminated, truncated, info)`

### 행동 공간 (Action Space)

* `spaces.Discrete(4)`
* 4개의 기술 중 하나를 선택:

  * 0: 몸통박치기(Tackle)
  * 1: 노려보기(Leer)
  * 2: 애교부리기(Charm)
  * 3: 꼬리흔들기(Tail Whip)

### 관측 공간 (Observation Space)

`spaces.Box(shape=(14,), low=0.0, high=1.0)`

정규화된 14차원 벡터

* 내 HP, 상대 HP (2)
* 내 방어력, 상대 방어력 (2)
* 내 각 기술 PP (4)
* 상대 각 기술 PP (4)
* 내 명중률 버프 누적값 (1)
* 상대 명중률 디버프 누적값 (1)

### 보상 설계 (Reward)

* 매 턴(step)마다:

  * `reward_step = (damage_to_opp - damage_to_me) / 100.0`
* 에피소드 종료 시:

  * 내가 승리: `+1.0`
  * 내가 패배: `-1.0`
  * HP가 남은 상태로 턴 제한 도달 등: `-0.2` (무승부 느낌)

### 에피소드 종료 조건

* 내 HP ≤ 0 또는 상대 HP ≤ 0 → `terminated = True`
* 턴 수가 최대 턴(기본 100)에 도달 → `truncated = True`

---

## 3. 학습 방법

### 3-1. DQN 학습

```bash
python train_dqn.py
```

`train_dqn.py` 주요 내용:

* `PokemonBattleEnv(render_mode=None)`으로 환경 생성

* `Monitor` 래퍼로 에피소드 리워드/길이 기록

* Stable-Baselines3 DQN으로 학습:

  ```python
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
      ...
  )
  ```

* `EvalCallback`을 사용해 주기적으로 평가하고 **best model** 저장:

  * `./models/dqn_pokemon_battle_best/best_model.zip`

* 학습 종료 시 최종 모델:

  * `./models/dqn_pokemon_battle_final.zip`

* 학습 로그(TensorBoard):

  * `./logs/dqn_pokemon_battle/`

### 3-2. PPO 학습

```bash
python train_ppo.py
```

`train_ppo.py` 주요 내용:

* DQN과 동일한 환경을 사용하되, 알고리즘만 PPO로 변경:

  ```python
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
      ...
  )
  ```

* 평가 콜백으로 best model 저장:

  * `./models/ppo_pokemon_battle_best/best_model.zip`

* 최종 모델:

  * `./models/ppo_pokemon_battle_final.zip`

* 학습 로그:

  * `./logs/ppo_pokemon_battle/`

---

## 4. TensorBoard로 학습 곡선 확인

DQN/PPO 학습을 돌린 뒤, 다음 명령으로 TensorBoard를 실행합니다.

```bash
tensorboard --logdir ./logs
```

브라우저에서 `http://localhost:6006` 접속 후:

* `rollout/ep_rew_mean` : 에피소드 평균 리워드
* `rollout/ep_len_mean` : 에피소드 평균 길이(턴 수)
* `eval/mean_reward` : 평가용 에피소드 평균 리워드 (EvalCallback 사용 시)
* DQN vs PPO 곡선을 비교해 학습 속도, 안정성, 최종 성능을 분석할 수 있습니다.

---

## 5. 플레이 / 평가 스크립트

### 5-1. 랜덤 정책 성능 비교

```bash
python play_random.py
```

* 에이전트는 `env.action_space.sample()`로 **완전 랜덤 행동**을 선택합니다.
* 여러 에피소드를 돌려:

  * 평균 리워드
  * 승/패/무 비율
    을 출력하여 RL 에이전트의 baseline으로 사용합니다.

### 5-2. DQN 학습 모델 플레이

```bash
python play_dqn_result.py
```

* `./models/dqn_pokemon_battle_best/best_model`을 로드해 평가합니다.
* 기본적으로 `render_mode="human"`으로 파이게임 창을 띄워 배틀을 시각화합니다.
* GUI 환경이 없거나 빠르게 성능만 보고 싶다면:

  * `PokemonBattleEnv(render_mode="human")` → `render_mode=None`으로 변경하면 터미널 로그만 출력됩니다.

### 5-3. PPO 학습 모델 플레이

```bash
python play_ppo_result.py
```

* `./models/ppo_pokemon_battle_best/best_model`을 로드해 평가합니다.
* 마찬가지로 파이게임 창에서 배틀을 확인하거나, `render_mode=None`으로 설정해 CLI 출력만 볼 수 있습니다.
* 스크립트에서는 여러 에피소드를 돌리며:

  * 에피소드별 총 리워드
  * 승/패/무 횟수
  * 승률
    을 요약해서 출력합니다.

---

## 6. 디렉터리 구조 예시

프로젝트 구조

```text
project_root/
├─ pokemon_battle_env.py        # Gymnasium 스타일 환경 클래스
├─ battle_step.py               # step() 내부 로직 분리
├─ pygame_render.py             # Pygame 렌더링 로직
├─ train_dqn.py                 # DQN 학습 스크립트
├─ train_ppo.py                 # PPO 학습 스크립트
├─ play_random.py               # 랜덤 정책 평가
├─ play_dqn_result.py           # DQN best_model 플레이/평가
├─ play_ppo_result.py           # PPO best_model 플레이/평가
├─ requirements.txt
├─ assets/
│   ├─ Bulbasaur.png
│   └─ Snorlax.png
└─ models/
    ├─ dqn_pokemon_battle_best/
    │   └─ best_model.zip
    ├─ ppo_pokemon_battle_best/
    │   └─ best_model.zip
    ├─ dqn_pokemon_battle_final.zip
    └─ ppo_pokemon_battle_final.zip
```


까지 한 번에 따라 해볼 수 있을 거야.
혹시 실제 코드 구조랑 다른 부분이 생기면, 파일명/폴더명만 살짝 맞춰서 쓰면 된다.
