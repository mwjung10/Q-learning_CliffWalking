import gymnasium as gym
from q_algorithm import QLearningAgent


if __name__ == "__main__":
    env = gym.make("CliffWalking-v0", render_mode="ansi")
    observation, info = env.reset()

    print(env.render())

    print("Action Space {}".format(env.action_space))
    print("State Space {}".format(env.observation_space))

    agent = QLearningAgent(
        env
    )

    agent.train(max_episodes=500)
