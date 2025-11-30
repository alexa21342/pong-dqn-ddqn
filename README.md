# pong-dqn-ddqn
Deep Q-Learning and Double DQN on Atari Pong for CSCI 166.
# Deep Q-Learning and Double DQN on Atari Pong

Author: **Alex Cortez**  
Course: **CSCI 166**

---

## Project Overview

This project implements a baseline **Deep Q-Network (DQN)** and an improved **Double DQN (DDQN)** agent on the Atari **Pong** environment using Gymnasium (`ALE/Pong-v5`).

Goals:

- Train a baseline DQN agent from raw pixels
- Implement Double DQN as a small algorithmic change
- Compare learning curves and final performance
- Record early vs late gameplay videos

---

## Environment & Observations

- **Environment:** `ALE/Pong-v5` (Gymnasium + ALE)
- **Observation preprocessing:**
  - RGB frames → grayscale
  - Resized to **84×84**
  - **4-frame stack** → final input shape `(4, 84, 84)`
- **Action space:** 6 discrete actions  
  (only *UP*, *DOWN*, and *NOOP* matter for actual gameplay)
- **Rewards:**  
  - `+1` when the agent scores  
  - `-1` when the opponent scores  
  - `0` otherwise (sparse reward)

---

## Model Architecture

Both DQN and Double DQN use the same convolutional network:

- **Convolutional encoder**
  - Conv2d(4, 32, kernel_size=8, stride=4) + ReLU  
  - Conv2d(32, 64, kernel_size=4, stride=2) + ReLU  
  - Conv2d(64, 64, kernel_size=3, stride=1) + ReLU  
  - Flatten
- **Fully-connected head**
  - Linear(3136, 512) + ReLU  
  - Linear(512, 6) → Q-values for each action

Training details:

- Experience replay buffer
- Target network (periodically synced)
- ε-greedy exploration
- Adam optimizer with MSE loss

---

## Hyperparameters

| Component                 | Value                      |
|--------------------------|----------------------------|
| Learning rate            | 1e-4                       |
| Discount factor (γ)      | 0.99                       |
| Replay buffer size       | 10,000                     |
| Batch size               | 32                         |
| Target sync frequency    | every 500 frames           |
| Epsilon schedule         | 1.0 → 0.01 over 10,000 frames |
| Frame stack              | 4                          |
| Environment              | `ALE/Pong-v5`              |
| Baseline DQN episodes    | ~450                       |
| Double DQN episodes      | ~530                       |

---

## Double DQN vs DQN

- **Standard DQN target:**

  \[
  y = r + \gamma \max_{a'} Q_{\text{target}}(s', a')
  \]

- **Double DQN target:**

  1. Use the **online** network to select the best next action:

     \[
     a^* = \arg\max_{a'} Q_{\text{online}}(s', a')
     \]

  2. Use the **target** network to evaluate that action:

     \[
     y = r + \gamma Q_{\text{target}}(s', a^*)
     \]

This decoupling reduces Q-value overestimation and generally gives more stable learning.

---

## Learning Curves

Overall training performance (rolling mean of episodic returns):

![Pong DQN vs DDQN](figures/pong_dqn_ddqn_learning_curve.png)

- **DQN (baseline):**
  - Starts near **−21**
  - Gradually improves to around **−10 to −11**
  - Curve is noisy with noticeable instability

- **Double DQN:**
  - Learns faster in the first few hundred episodes
  - Reaches around **−11** and stays more stable
  - Gameplay appears smoother and more controlled

---

## Files in This Repository

- `notebook/pong_dqn_ddqn.ipynb`  
  Final Colab notebook with:
  - Environment setup & wrappers  
  - DQN + Double DQN training loops  
  - Logging and plotting code

- `report/Deep_Q-Learning_and_Double_DQN_on_Atari_Pong_Report.pdf`  
  3–5 page written report (setup, model, experiments, results, reflection).

- `slides/Deep_Q-Learning_Pong_Slides.pptx`  
  Slide deck summarizing the project.

- `figures/pong_dqn_ddqn_learning_curve.png`  
  Combined learning-curve plot used in the report and slides.

- `videos/pong_early_random.mp4`  
  Early training video (random policy / untrained agent).

- `videos/pong_late_ddqn.mp4`  
  Late training video (trained Double DQN agent).

- `models/*.dat` *(optional)*  
  Saved PyTorch weights for best baseline and DDQN models.

---

## How to Run (High Level)

1. Open `notebook/pong_dqn_ddqn.ipynb` in Google Colab.
2. Install dependencies at the top of the notebook:
   - `gymnasium[atari,accept-rom-license]`
   - `autorom`
   - `stable-baselines3`
   - `torch`, `numpy`, etc.
3. Run the **baseline DQN** training cell.
4. Run the **Double DQN** training cell.
5. Run the **plotting** cell to regenerate the learning curves.
6. (Optional) Run the **video-recording** cells to produce new early/late videos.

---

## Reflection (Short)

Pong was a good testbed because it is visually interpretable and has sparse rewards. The baseline DQN improved slowly and showed unstable learning, while Double DQN—changing only the target calculation—learned faster and more smoothly. This project showed how a small change in the algorithm (separating action selection from evaluation) can noticeably affect the stability and performance of deep reinforcement learning, even when the network architecture and environment stay the same.
