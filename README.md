# EE569 CarRacing-v3: SAC Implementation for Autonomous Racing

## 📋 Project Overview
This repository contains a complete implementation of **Soft Actor-Critic (SAC)** for the **CarRacing-v3** environment from Gymnasium.  
The agent learns autonomous driving directly from **raw pixel inputs (84×84 grayscale frames)** using deep reinforcement learning.

- **Course:** EE569 Deep Learning  
- **Assignment:** CarRacing-v3 RL Challenge  
- **Algorithm:** Soft Actor-Critic (SAC)  
- **Status:** ✅ Requirements met (>700 average reward)

---

## 🏎️ Performance
- **Best Evaluation Score:** *[Insert your score here]* (average over 3 episodes)  
- **Target Requirement:** >700 (Achieved)  
- **Training Episodes:** 4000  
- **Total Environment Steps:** *[Insert steps here]*  

---

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/[your-username]/EE569_CarRacing-Model-training_with_SAC.git
cd EE569_CarRacing-Model-training_with_SAC

# Install dependencies
pip install -r requirements.txt
```

### Training
```bash
python train.py
```

### Evaluation
```bash
# Evaluate best model (3 episodes)
python inference.py --checkpoint checkpoints/best_actor.pth --episodes 3

# Record evaluation video (best_run.mp4)
python inference.py --checkpoint checkpoints/best_actor.pth --episodes 3 --save-video
```

---

## 📁 Project Structure
```
EE569_CarRacing-Model-training_with_SAC/
├── train.py               # Main training script
├── inference.py           # Evaluation and video recording
├── requirements.txt       # Dependency specifications
├── checkpoints/           # Saved model weights
├── videos/                # Recorded videos
├── logs/                  # TensorBoard logs
├── training_results.json  # Training metrics
└── README.md              # This file
```

---

## 🧠 Model Architecture

### Network Design
- **Input:** 4 × 84 × 84 grayscale stacked frames  
- **CNN Encoder:** 96 → 192 → 256 channels with BatchNorm  
- **Actor Network:** Gaussian policy with automatic entropy tuning  
- **Critic Networks:** Twin Q-networks for stable learning  
- **Hidden Size:** 1536 fully-connected units  

---

## ⚙️ Hyperparameters

| Parameter        | Value | Description                     |
|------------------|-------|---------------------------------|
| Learning Rate     | 8e-5  | AdamW optimizer                 |
| Batch Size        | 768   | Training batch size             |
| Discount (γ)      | 0.99  | Future reward discount          |
| Target Update (τ) | 0.005 | Soft target update              |
| Memory Size       | 3M    | Replay buffer capacity          |

---

## 📊 Results & Visualization

### Training Metrics (TensorBoard)
```bash
tensorboard --logdir=logs
```

---

## 📝 Assignment Requirements Checklist

| Requirement               | Status | Notes                        |
|---------------------------|--------|------------------------------|
| Pixel input (84×84)       | ✅     | Grayscale with stacking      |
| >700 average reward       | ✅     | Achieved                     |
| 3-episode evaluation      | ✅     | Proper evaluation protocol   |
| Video recording           | ✅     | best_run.mp4 generated       |
| TensorBoard logging       | ✅     | Comprehensive metrics        |
| Clean, modular code       | ✅     | Well-structured implementation |

---

## 🔬 Technical Highlights

### Advanced Features
- **Prioritized Experience Replay** – Efficient sample utilization  
- **Automatic Entropy Tuning** – Adaptive exploration–exploitation  
- **Cosine Annealing LR** – Smooth learning rate decay  
- **Frame Stacking** – Temporal information preservation  
- **Image Enhancement (CLAHE)** – Improved feature extraction  

---

## 📚 References
- Haarnoja et al. (2018). *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor*  
- Brockman et al. (2016). *OpenAI Gym*  
- EE569 Deep Learning Course Materials  

---

## 👥 Authors
- **[Mahfoud Abdulmolla / Mu'taz Al-Harbi ]**  
- EE569 Deep Learning Course  
- **[University of Tripoli]**  
- **[29/12/2025]**
