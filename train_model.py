from TicTacToe import TicTacToe
import numpy as np
import random
import os
from collections import deque
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

# Hyperparameters for training 
BATCH_SIZE = 32
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_DECAY = 0.999
EPSILON_MIN = 0.1
EPISODES = 5000
MODEL_PATH = '2022AAPS0419G_MODEL.h5'

def build_model():
    model = Sequential()
    model.add(Dense(64, input_dim=9, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(9, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

def load_or_initialize_model():
    if os.path.exists(MODEL_PATH):
        print("Loading existing model")
        return load_model(MODEL_PATH)
    else:
        print("Creating new model")
        return build_model()

def state_change(state):
    return np.array([-1 if x == 2 else x for x in state])

def train_sqn():
    print("Starting AI training...")
    print(f"Training for {EPISODES} episodes...")
    
    model = load_or_initialize_model()
    replay_buffer = deque(maxlen=10000)
    episode_buffer = []
    epsilon = EPSILON_START
    wins_sq, wins_p1, draws = 0, 0, 0

    for episode in range(EPISODES):
        game = TicTacToe(0)
        state = state_change(game.board)
        done = False
        episode_buffer.clear()
        
        while not done:
            game.player1_move()

            if game.is_full() or game.current_winner == 1:
                if game.current_winner == 1:
                    wins_p1 += 1
                else:
                    draws += 1
                break

            state = state_change(game.board)
            valid_actions = [i for i, x in enumerate(state) if x == 0]
            
            if np.random.rand() <= epsilon:
                action = random.choice(valid_actions)
            else:
                q_values = model.predict(state[np.newaxis, :], verbose=0)[0]
                action = max(valid_actions, key=lambda a: q_values[a])

            game.make_move(action, 2)
            reward = game.get_reward()
            next_state = state_change(game.board)
            done = game.is_full() or game.current_winner is not None

            episode_buffer.append((state, action, reward, next_state, done))
            state = next_state

            if done:
                if game.current_winner == 2:
                    wins_sq += 1
                elif game.is_full() and game.current_winner is None:
                    draws += 1
                break

        replay_buffer.extend(episode_buffer)

        if episode % 2 == 1 and len(replay_buffer) >= BATCH_SIZE:
            mini_batch = random.sample(replay_buffer, BATCH_SIZE)
            states = np.zeros((BATCH_SIZE, 9))
            q_values_batch = np.zeros((BATCH_SIZE, 9))

            for j, (s, a, r, s_next, d) in enumerate(mini_batch):
                q_values = model.predict(s[np.newaxis, :], verbose=0)[0]
                q_next = model.predict(s_next[np.newaxis, :], verbose=0)[0]

                if d:
                    q_values[a] = r
                else:
                    valid_next_actions = [i for i, x in enumerate(s_next) if x == 0]
                    q_values[a] = r + GAMMA * max([q_next[a] for a in valid_next_actions])

                states[j] = s
                q_values_batch[j] = q_values

            model.fit(states, q_values_batch, epochs=1, verbose=0)

        if episode % 2 == 1:
            epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

        if (episode + 1) % 100 == 0:
            win_rate_sq = (wins_sq / (episode + 1)) * 100
            print(f"Episode {episode + 1}: AI Win Rate = {win_rate_sq:.2f}%")

    model.save(MODEL_PATH)
    print(f"\n✅ Training completed!")
    print(f"Model saved as '{MODEL_PATH}'")

    total_episodes = EPISODES
    win_rate_sq = (wins_sq / total_episodes) * 100
    print(f"Final AI Win Rate: {win_rate_sq:.2f}%")
    print(f"Total Wins - AI: {wins_sq}, Player 1: {wins_p1}, Draws: {draws}")

if __name__ == "__main__":
    train_sqn() 