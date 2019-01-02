import gym
env = gym.make('Pendulum-v0')
for i in range(100):
    env.reset()
    env.render()
    action = env.action_space.sample()
    state = env.step(action)
    print(state)
    print(env.action_space.shape[0])
    print(env.observation_space.shape[0])