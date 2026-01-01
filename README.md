## 🏎️ CarRacing-v3 SAC Agent (Optimized for RTX 4050 Laptop)

This project implements a **Soft Actor-Critic (SAC)** agent to solve the `CarRacing-v3` environment. It has been heavily optimized for consumer hardware (specifically **6GB VRAM / 16GB RAM**) to achieve maximum FPS and stability without crashing.

## ⚡ Key Optimizations (The "Maximized Pipeline")

We achieved **~9 FPS** (up from 2 FPS) and stable training on an RTX 4050 Laptop GPU by implementing the following:

### 1. Memory Management (RAM & VRAM)
*   **Uint8 Buffer:** The Replay Buffer stores states as `uint8` (0-255) instead of `float32`, reducing RAM usage by **4x**. Normalization happens on the GPU on-the-fly.
*   **Pre-allocated Numpy Arrays:** We use a custom `buffer.py` with pre-allocated arrays to prevent memory fragmentation and garbage collection spikes.
*   **Reduced Hidden Size:** `HIDDEN_SIZE = 256` (down from 512) to fit within 6GB VRAM while maintaining learning capacity.
*   **PyTorch Allocation:** `os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"` prevents "CUDA out of memory" fragmentation errors.

### 2. Speed & Throughput
*   **Async Data Transfer:** `non_blocking=True` is used when moving batches to GPU, allowing data transfer to overlap with computation.
*   **Torch Compile:** `torch.compile()` is enabled for the Actor and Critic networks (PyTorch 2.0+ feature).
*   **Update Frequency:** We update the network every **16 steps** (instead of 1 or 8).
*   **Reduced Update Ratio:** We perform `update_freq // 4` updates per cycle (0.25 updates per step). This significantly reduces the GPU computational load while maintaining sample efficiency.
*   **Action Repeat:** `ACTION_REPEAT = 4` (Frame skipping) allows the agent to see further into the future and train 4x faster.

---

## 🚀 Quick Start

### 1. Setup
```bash
# 1. Create virtual environment with Python 3.12
uv venv --python 3.12

# 2. Activate environment
source .venv/bin/activate

# 3. Install dependencies
uv pip install swig
uv pip install -r requirements.txt
```

### 2. Train (Optimized Command)
Run the training with the specific batch size that fits 6GB VRAM:

```bash
uv run train.py --batch-size 256
```

*   **Batch Size 256:** The sweet spot. 400+ risks OOM on 6GB cards.
*   **Update Freq 16:** Default in code, ensures high FPS.

### 3. Monitor
```bash
tensorboard --logdir=logs
```
Open `http://localhost:6006` to see rewards, losses, and entropy.

---

## ⚙️ Configuration Guide

If you need to tweak settings for different hardware, here is where to look in `train.py`:

### Hardware Profiles

| Hardware | VRAM | RAM | Recommended Settings |
| :--- | :--- | :--- | :--- |
| **RTX 4050 (Current)** | **6GB** | **16GB** | `BATCH_SIZE=256`, `MEMORY_SIZE=200000`, `HIDDEN_SIZE=256` |
| **RTX 3060 / 4060** | **8GB** | **16GB** | `BATCH_SIZE=512`, `MEMORY_SIZE=200000`, `HIDDEN_SIZE=256` |
| **RTX 3080 / 4080** | **12GB+** | **32GB** | `BATCH_SIZE=1024`, `MEMORY_SIZE=500000`, `HIDDEN_SIZE=512` |

### Key Constants in `train.py`

*   **Line 28:** `BATCH_SIZE` - Controls VRAM usage. Lower this if you get CUDA OOM.
*   **Line 33:** `MEMORY_SIZE` - Controls RAM usage. `200000` uses ~13GB system RAM. Lower to `150000` if you have background apps open.
*   **Line 34:** `HIDDEN_SIZE` - Network width. `256` is sufficient for CarRacing.
*   **Line 505:** `parser.add_argument('--update-freq', ...)` - Controls speed vs sample efficiency. Higher = Faster FPS, Slower convergence.

---

## ⏸️ Pause & Resume

*   **Pause:** Press `Ctrl+C` at any time. The agent will safely save the checkpoint and the replay buffer.
*   **Resume:** Simply run the training command again.
    *   It automatically detects `checkpoints/latest.pth`.
    *   It automatically loads `checkpoints/latest_buffer.npz`.
    *   It fixes "Stuck Alpha" issues automatically.

## 📁 Project Structure

*   `train.py`: Main training loop, SAC implementation, and hyperparameters.
*   `buffer.py`: Optimized Numpy-based Replay Buffer (Critical for RAM performance).
*   `checkpoints/`: Stores `.pth` models and `.npz` buffers.
*   `videos/`: Stores evaluation replays (MP4/GIF) to see how the agent drives.
*   `logs/`: Tensorboard logs.
