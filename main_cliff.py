import gymnasium as gym
from q_algorithm import QLearningAgent
import pandas as pd
from tabulate import tabulate


def average_experiments(exp_prob=0.1):
    env = gym.make("CliffWalking-v0", render_mode="ansi")
    agent = QLearningAgent(env)

    learning_rates = [0.01, 0.1, 0.5, 1.0]
    episodes_list = [100, 500, 1000, 2000]
    n_runs = 25

    results = []

    for lr in learning_rates:
        for episodes in episodes_list:
            avg_success = 0
            avg_fail = 0
            avg_steps = 0
            avg_last10 = 0

            for _ in range(n_runs):
                env = gym.make("CliffWalking-v0")
                agent = QLearningAgent(
                    env,
                    learning_rate=lr,
                    exploration_prob=exp_prob
                )

                metrics = agent.train(max_episodes=episodes)
                avg_success += metrics["success_rate"]
                avg_fail += metrics["cliff_fail_rate"]
                avg_steps += metrics["avg_step_count"]
                avg_last10 += metrics["rewards_last_10"]
                env.close()

            avg_success /= n_runs
            avg_fail /= n_runs
            avg_steps /= n_runs
            avg_last10 /= n_runs

            results.append({
                "Learning rate": lr,
                "Episodes": episodes,
                "Success rate": f"{avg_success:.1%}",
                "Cliff fail rate": f"{avg_fail:.1%}",
                "Avg steps": f"{avg_steps:.1f}",
                "Avg last 10 rewards ": f"{avg_last10:.1f}"
            })

    df = pd.DataFrame(results)
    df = df.sort_values(by=["Learning rate", "Episodes"])
    print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))

    return df


if __name__ == "__main__":
    env = gym.make("CliffWalking-v0", render_mode="ansi")
    observation, info = env.reset()

    exploration_prob = [0.01, 0.05, 0.1, 0.5]
    for prob in exploration_prob:
        print(f"\nExploration Probability: {prob}")
        average_experiments(exp_prob=prob)
