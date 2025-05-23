import numpy as np


class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, initial_exploration_prob=1.0, exploration_decay=0.99,
                 min_exploration_prob=0.01):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = initial_exploration_prob
        self.exploration_decay = exploration_decay
        self.min_exploration_prob = min_exploration_prob
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))

    def _choose_action(self, state):
        if np.random.rand() < self.exploration_prob:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_table[state])

    def train(self, max_episodes=100):
        for episode in range(max_episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                action = self._choose_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                total_reward += reward

                best_next_action = np.argmax(self.q_table[next_state])
                td_target = reward + self.discount_factor * self.q_table[next_state, best_next_action]
                td_error = td_target - self.q_table[state, action]
                self.q_table[state, action] += self.learning_rate * td_error
                state = next_state

            if episode % 100 == 0:
                print(f"Episode: {episode}, Reward: {total_reward}, Exploration: {self.exploration_prob:.3f}")

            self.exploration_prob = max(
                self.min_exploration_prob,
                self.exploration_prob * self.exploration_decay
            )

