import os
import random
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

# Function to preprocess the board state
def state_change(state):
    return np.array([-1 if x == 2 else x for x in state])

class PlayerSQN:
    def __init__(self, epsilon=0.01):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "2022AAPS0419G_MODEL.h5") 
        self.model = load_model(model_path, custom_objects={"mse": MeanSquaredError()})
        self.epsilon = epsilon

    def move(self, state):
        preprocessed_state = state_change(state)
        state_array = preprocessed_state.reshape(1, -1)

        if random.random() < self.epsilon:
            valid_actions = [i for i, x in enumerate(state) if x == 0]
            action = random.choice(valid_actions)
        else:
            q_values = self.model.predict(state_array, verbose=0)[0]
            valid_actions = [i for i, x in enumerate(preprocessed_state) if x == 0]
            action = max(valid_actions, key=lambda x: q_values[x])

        # Debug prints can be commented out for production
        # print(f"Valid actions: {valid_actions}")
        # print(f"Final chosen action: {action}")
        return action 