# -*- coding: utf-8 -*-
import gym
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

if __name__ == "__main__":
    env = gym.make("MountainCar-v0")
    state_size  = env.observation_space.shape[0]
    model   = Sequential()
    model.add(Dense(24, input_dim=state_size, activation="relu"))
    model.add(Dense(48, activation="relu"))
    model.add(Dense(24, activation="relu"))
    model.add(Dense(env.action_space.n))
    model.compile(loss="mean_squared_error", optimizer=Adam(lr=0.001))
    model.load_weights("success.model")
    state = env.reset()
    state = np.reshape(state, [1,state_size])
    while  True:
        env.reset()
        for time in range(500):
            env.render()
            action = model.predict(state)
            next_state, reward, done, _ = env.step(np.argmax(action[0]))
            next_state = np.reshape(next_state, [1,state_size])
            state = next_state