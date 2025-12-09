# Pokemon Reinforcement Learning Battle

단순화된 포켓몬 1:1 배틀 환경(`PokemonBattleEnv`)에서 강화학습 에이전트(DQN, PPO)를 학습·평가하는 프로젝트입니다.

- 환경: **Gymnasium 호환** 포켓몬 배틀 환경
- 알고리즘: **Stable-Baselines3** 의 **DQN, PPO**
- 렌더링: **Pygame** 으로 실제 포켓몬 대전처럼 시각화

---

## 1. 실행 환경 & 설치

### Python 버전

- `pyenv` 를 사용하여 **Python 3.12.3** 환경에서 개발 및 테스트했습니다.
- 3.10.x 버전에서 돌아가지 않는 패키지 버전이 있어 다운그레이드하여 requirements.txt 파일을 다시 푸쉬하였습니다.

### 의존성 설치

```
pip install -r requirements.txt
````

`requirements.txt`에는 대략 다음 패키지가 포함되어 있습니다.

* `stable-baselines3`
* `gymnasium`
* `numpy`
* `pygame`
* `torch` (CPU만으로도 실행 가능)
* `tensorboard` (학습 곡선 시각화용)

> **참고:** 이번 프로젝트에서는 CUDA/GPU를 사용하지 않고 CPU 버전 PyTorch로만 실행했습니다.

### 에셋(이미지) 준비

Pygame 렌더링에서 `assets/` 폴더의 이미지를 사용합니다.

* `assets/Bulbasaur.png`
* `assets/Snorlax.png`

이미지 파일 경로나 파일명을 변경할 경우, `pygame_render.py` 내부의 이미지 로드 경로도 함께 수정해야 합니다.

---

## 2. 환경 개요 (`PokemonBattleEnv`)

* 파일: `pokemon_battle_env.py`
* Gymnasium 스타일 인터페이스:

  * `reset() -> (obs, info)`
  * `step(action) -> (obs, reward, terminated, truncated, info)`

### 행동 공간 (Action Space)

* `spaces.Discrete(4)`
* 4개의 기술 중 하나를 선택합니다.

  * `0`: 몸통박치기 (Tackle)
  * `1`: 노려보기 (Leer)
  * `2`: 애교부리기 (Charm)
  * `3`: 꼬리흔들기 (Tail Whip)

### 관측 공간 (Observation Space)

* `spaces.Box(shape=(14,), low=0.0, high=1.0)`
* 정규화된 14차원 실수 벡터로 구성됩니다.

  * 내 HP, 상대 HP (2)
  * 내 방어력, 상대 방어력 (2)
  * 내 각 기술 PP (4)
  * 상대 각 기술 PP (4)
  * 내 명중률 버프 누적값 (1)
  * 상대 명중률 디버프 누적값 (1)

### 보상 설계 (Reward)

* 매 턴(step)마다:

  ```text
  reward_step = (damage_to_opp - damage_to_me) / 100.0
  ```

* 에피소드 종료 시:

  * 내가 승리: `+1.0`
  * 내가 패배: `-1.0`
  * 둘 다 살아 있고 턴 제한 도달 등: `-0.2` (무승부 패널티)

### 에피소드 종료 조건

* 내 HP ≤ 0 또는 상대 HP ≤ 0 → `terminated = True`
* 턴 수가 최대 턴(기본 100)에 도달 → `truncated = True`

---

## 3. 학습 방법

### 3-1. DQN 학습

```
python train_dqn.py
```

`train_dqn.py`는 다음을 수행합니다.

* `PokemonBattleEnv(render_mode=None)` 으로 환경 생성
* `Monitor` 래퍼로 에피소드 리워드/길이 기록
* Stable-Baselines3 **DQN** 으로 학습 진행
* `EvalCallback` 으로 주기적 평가 및 best model 저장

저장 경로:

* 최고 성능 모델(best model):

  ```text
  ./models/dqn_pokemon_battle_best/best_model.zip
  ```

* 최종 모델(final model):

  ```text
  ./models/dqn_pokemon_battle_final.zip
  ```

* 학습 로그(TensorBoard):

  ```text
  ./logs/dqn_pokemon_battle/
  ```

### 3-2. PPO 학습

```
python train_ppo.py
```

`train_ppo.py`는 다음을 수행합니다.

* 동일한 `PokemonBattleEnv` 를 사용하고, 알고리즘만 **PPO** 로 변경
* `Monitor` 래퍼로 로그 기록
* `EvalCallback` 으로 평가 및 best model 저장

저장 경로:

* 최고 성능 모델(best model):

  ```text
  ./models/ppo_pokemon_battle_best/best_model.zip
  ```

* 최종 모델(final model):

  ```text
  ./models/ppo_pokemon_battle_final.zip
  ```

* 학습 로그(TensorBoard):

  ```text
  ./logs/ppo_pokemon_battle/
  ```

---

## 4. TensorBoard로 학습 곡선 확인

DQN / PPO 학습 후, 다음 명령으로 TensorBoard를 실행합니다.

```
tensorboard --logdir ./logs
```

브라우저에서 `http://localhost:6006` 접속 후, 주요 지표를 확인할 수 있습니다.

* `rollout/ep_rew_mean` : 에피소드 평균 리워드
* `rollout/ep_len_mean` : 에피소드 평균 길이(턴 수)
* `eval/mean_reward` : 평가용 에피소드 평균 리워드 (EvalCallback 사용 시)

DQN과 PPO의 곡선을 겹쳐보면서 학습 속도, 수렴 안정성, 최종 성능을 비교할 수 있습니다.

---

## 5. 플레이 / 평가 스크립트

### 5-1. 랜덤 정책 성능 비교

```
python play_random.py
```

* 에이전트는 `env.action_space.sample()` 로 **완전 랜덤 행동**을 선택합니다.
* 여러 에피소드를 돌려:

  * 평균 리워드
  * 승 / 패 / 무 비율
    을 출력하며, DQN/PPO와 비교할 **baseline**으로 사용합니다.

### 5-2. DQN 학습 모델 플레이

```
python play_dqn_result.py
```

* `./models/dqn_pokemon_battle_best/best_model` 을 로드해 평가합니다.
* 기본적으로 `render_mode="human"` 으로 Pygame 창을 띄워 배틀을 시각화합니다.

> GUI 환경이 없거나, 속도만 보고 싶다면
> `PokemonBattleEnv(render_mode="human")` → `render_mode=None` 으로 변경하여
> 터미널 로그만 출력되도록 설정할 수 있습니다.

### 5-3. PPO 학습 모델 플레이

```
python play_ppo_result.py
```

* `./models/ppo_pokemon_battle_best/best_model` 을 로드해 평가합니다.
* Pygame 창에서 실제 포켓몬 배틀처럼 애니메이션을 확인할 수 있습니다.
* 스크립트는 여러 에피소드를 돌리며:

  * 에피소드별 총 리워드
  * 승 / 패 / 무 횟수
  * 승률
    을 요약해서 출력합니다.

---

## 6. 디렉터리 구조 예시

프로젝트 구조는 대략 다음과 같습니다.

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
