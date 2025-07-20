# TicTacToe RL: Play Against a Trained AI 

Welcome! This project lets you play TicTacToe against an AI trained with Q-Learning and a Shallow Q-Network (SQN). The AI has learned to play optimally through thousands of games—can you beat it?

---

## How the AI Learns (Q-Learning + SQN)

This project is a hands-on demonstration of **Reinforcement Learning** using:
- **Q-Learning**: A classic RL algorithm that learns the value of actions in each state.
- **Shallow Q-Network (SQN)**: A simple neural network that approximates the Q-value function, allowing the AI to generalize across similar board states.

### Q-Learning Algorithm (used here)
At each step, the agent:
1. Observes the current state `s`
2. Chooses an action `a` (explore or exploit)
3. Receives a reward `r` and observes the next state `s'`
4. Updates its Q-value estimate:

```python
Q(s, a) = Q(s, a) + lr * (r + gamma * max(Q(s', a')) - Q(s, a))
```
- `lr`: learning rate
- `gamma`: discount factor
- `max(Q(s', a'))`: best future value from next state

### Shallow Q-Network (SQN)
Instead of a table, we use a neural network:
- **Input**: 9 board cells (flattened)
- **Hidden layers**: 2 layers, 64 neurons each, ReLU activation
- **Output**: 9 Q-values (one for each possible move)

The network is trained to minimize the difference between predicted Q-values and the target Q-values from the Q-learning update.

### Training Details
- **Episodes**: 5000 games of self-play
- **Experience Replay**: Learns from a buffer of past moves for stability
- **Epsilon-Greedy**: Starts exploring, gradually exploits more as it learns
- **Reward System**: +1 for win, -1 for loss, 0 for draw
- **Result**: ~95% win rate after training

---

##  Getting Started

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Train the AI (if you haven’t already)
```bash
python train_model.py
```
> This creates the file `2022AAPS0419G_MODEL.h5` (the AI’s brain).

### 3. Launch the Game
```bash
streamlit run streamlit_app.py
```
Open your browser to the link shown (usually [http://localhost:8501](http://localhost:8501)).

---

##  How to Play
- **You** are  (X) and always go first.
- **AI** is  (O) and responds instantly.
- Click any empty cell to make your move.
- First to get 3 in a row wins!
- See your stats and AI details in the sidebar.

---

## 📁 Project Structure
- `streamlit_app.py` — The web app
- `PlayerSQN.py` — The AI agent (SQN)
- `TicTacToe.py` — Game logic
- `2022AAPS0419G_MODEL.h5` — Trained model (auto-generated)
- `requirements.txt` — Dependencies
- `train_model.py` —  to train the AI

---

## Why This Project?
- **Showcases real RL in action** 


---

##  Ideas for the Future
- [ ] Add difficulty levels
- [ ] Visualize the AI’s learning process
- [ ] Try different board sizes
- [ ] Compare different RL algorithms

---