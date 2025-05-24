import numpy as np
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    handlers=[logging.StreamHandler()])

logging.disable(logging.INFO)


class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, exploration_prob=0.1):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
        self.q_table = np.zeros((env.observation_space.n, env.action_space.n))

        # for checking the accuracy
        self.success_rate = 0  # meaure how much the agent is successful
        self.rewards_last_10 = []  # store the last 10 rewards to check the performance
        self.cliff_fail_rate = 0  # measure how much the agent fails to reach the goal
        self.avg_step_count = 0  # average number of steps to reach the goal

    def _choose_action(self, state):
        if np.random.rand() < self.exploration_prob:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_table[state])

    def train(self, max_episodes=100):

        steps_array = []
        for episode in range(max_episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            steps = 0

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
                steps += 1

            steps_array.append(steps)

            if episode % 100 == 0:
                logging.info(f"Episode: {episode}, Reward: {total_reward}, Exploration: {self.exploration_prob:.3f}")


            # Update success rate, cliff fail rate, and average step count
            if total_reward >= -100:
                self.success_rate += 1
                if ((max_episodes - episode) < 10):
                    self.rewards_last_10.append(total_reward)
            else:
                self.cliff_fail_rate += 1
        # Calculate success rate, cliff fail rate, and average step count
        self.avg_step_count = np.mean(steps_array)
        self.success_rate /= max_episodes
        self.cliff_fail_rate /= max_episodes

        return {
            "success_rate": self.success_rate,
            "cliff_fail_rate": self.cliff_fail_rate,
            "avg_step_count": self.avg_step_count,
            "rewards_last_10": np.mean(self.rewards_last_10) if self.rewards_last_10 else 0
        }

