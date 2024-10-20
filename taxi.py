"""
Dans ce TP, nous allons implémenter un agent qui apprend à jouer au jeu Taxi-v3
de OpenAI Gym. Le but du jeu est de déposer un passager à une destination
spécifique en un minimum de temps. Le jeu est composé d'une grille de 5x5 cases
et le taxi peut se déplacer dans les 4 directions (haut, bas, gauche, droite).
Le taxi peut prendre un passager sur une case spécifique et le déposer à une
destination spécifique. Le jeu est terminé lorsque le passager est déposé à la
destination. Le jeu est aussi terminé si le taxi prend plus de 200 actions.

Vous devez implémenter un agent qui apprend à jouer à ce jeu en utilisant
les algorithmes Q-Learning et SARSA.

Pour chaque algorithme, vous devez réaliser une vidéo pour montrer que votre modèle fonctionne.
Vous devez aussi comparer l'efficacité des deux algorithmes en termes de temps
d'apprentissage et de performance.

A la fin, vous devez rendre un rapport qui explique vos choix d'implémentation
et vos résultats (max 1 page).
"""

import typing as t
import gymnasium as gym
import numpy as np
from qlearning import QLearningAgent
from qlearning_eps_scheduling import QLearningAgentEpsScheduling
from sarsa import SarsaAgent
from gymnasium.wrappers import RecordVideo
import os
import matplotlib.pyplot as plt

env = gym.make("Taxi-v3", render_mode="rgb_array")
n_actions = env.action_space.n  # type: ignore

def initialize_environment(enable_recording: bool = False, output_dir: str = "videos", record_episode: int = 0) -> gym.Env:
    """
    Initialize the Taxi-v3 environment with an optional video recording setup.
    
    Args:
        enable_recording (bool): Flag to enable or disable video recording.
        output_dir (str): Directory to save recorded videos.
        record_episode (int): Specifies which episode to record.
    
    Returns:
        gym.Env: The initialized environment.
    """
    # Create the Taxi-v3 environment
    environment = gym.make("Taxi-v3", render_mode="rgb_array")
    
    if enable_recording:
        # Ensure the output directory for videos exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Wrap the environment with video recording functionality
        environment = RecordVideo(
            environment,
            video_folder=output_dir,
            episode_trigger=lambda episode_id: episode_id == record_episode,
            name_prefix="taxi-agent"
        )
    
    return environment

env = initialize_environment(enable_recording=True, output_dir="records_q", record_episode=999)  # Record the 1000th episode

#################################################
# 1. Play with QLearningAgent
#################################################

agent = QLearningAgent(
    learning_rate=0.1, epsilon=1.0, gamma=0.99, legal_actions=list(range(n_actions))
)

def play_and_train(env: gym.Env, agent: QLearningAgent, t_max=int(1e4)) -> float:
    """
    This function should
    - run a full game, actions given by agent.getAction(s)
    - train agent using agent.update(...) whenever possible
    - return total reward
    """
    total_reward: t.SupportsFloat = 0.0
    s, _ = env.reset()

    step_count = 0
    done = False
    while not done and step_count < t_max:
        # Get agent to pick action given state s
        a = agent.get_action(s)

        next_s, r, done, _, _ = env.step(a)

        # Train agent for state s
        agent.update(s, a, r, next_s)
        total_reward += r
        s = next_s
        step_count += 1

    return total_reward

rewards_q = []
best_mean_reward = float('-inf')
no_improvement_count = 0
episode_count = 0
while episode_count < 10000:
    rewards_q.append(play_and_train(env, agent))
    if episode_count % 100 == 0:
        mean_reward = np.mean(rewards_q[-100:])
        print(f"[Épisode {episode_count}] - Récompense moyenne : {mean_reward:.2f} | Épsilon : {agent.epsilon:.4f}")

        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            print(f"🔝 Nouvelle meilleure récompense moyenne : {best_mean_reward:.2f}")
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if mean_reward > 0:
            print("✅ Performance satisfaisante atteinte!")
            break
        
        if no_improvement_count >= 50:  # Si pas d'amélioration pendant 5000 épisodes
            print("🔄 Réinitialisation de l'agent en raison d'un manque d'amélioration.")
            agent.reset()
            no_improvement_count = 0

    episode_count += 1

assert np.mean(rewards_q[-100:]) > 0.0

plt.plot(rewards_q,color='purple')
plt.xlabel('Épisode')
plt.ylabel('Total Reward')
plt.title('Q-Learning Agent Performance')
plt.show()
env.close()

#################################################
# 2. Play with QLearningAgentEpsScheduling
#################################################

env = initialize_environment(enable_recording=True, output_dir="records_q_eps", record_episode=999)  # Record the 1000th episode
agent = QLearningAgentEpsScheduling(
    learning_rate=0.1,
    epsilon=1.0,
    gamma=0.99,
    legal_actions=list(range(n_actions)),
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay_steps=100000  # Augmenter le nombre d'étapes pour la décroissance
)

rewards_q_eps = []
best_mean_reward = float('-inf')
no_improvement_count = 0
episode_count = 0
while episode_count < 10000:
    rewards_q_eps.append(play_and_train(env, agent))
    if episode_count % 100 == 0:
        mean_reward = np.mean(rewards_q_eps[-100:])
        print(f"[Épisode {episode_count}] - Récompense moyenne : {mean_reward:.2f} | Épsilon : {agent.epsilon:.4f}")

        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            print(f"🔝 Nouvelle meilleure récompense moyenne : {best_mean_reward:.2f}")
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if mean_reward > 0:
            print("✅ Performance satisfaisante atteinte!")
            break
        
        if no_improvement_count >= 50:  # Si pas d'amélioration pendant 5000 épisodes
            print("🔄 Réinitialisation de l'agent en raison d'un manque d'amélioration.")
            agent.reset()
            no_improvement_count = 0

    episode_count += 1

assert np.mean(rewards_q_eps[-100:]) > 0.0

# TODO: créer des vidéos de l'agent en action
env.close()
plt.plot(rewards_q_eps, color='orange')
plt.xlabel('Épisode')
plt.ylabel('Total Reward')
plt.title('Q-Learning Agent with Epsilon Scheduling')
plt.show()

######################
# 3. Play with SARSA
######################

env = initialize_environment(enable_recording=True, output_dir="records_sarsa", record_episode=999)  # Record the 1000th episode

agent = SarsaAgent(learning_rate=0.1, gamma=0.99, legal_actions=list(range(n_actions)))

rewards_sarsa = []
episode_count = 0
while episode_count < 10000:
    rewards_sarsa.append(play_and_train(env, agent))
    if episode_count % 100 == 0:
        mean_reward = np.mean(rewards_sarsa[-100:])
        print(f"[Épisode {episode_count}] - Récompense moyenne : {mean_reward:.2f}")

        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            print(f"🔝 Nouvelle meilleure récompense moyenne : {best_mean_reward:.2f}")
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if mean_reward > 0:
            print("✅ Performance satisfaisante atteinte!")
            break
        
        if no_improvement_count >= 50:  # Si pas d'amélioration pendant 5000 épisodes
            print("🔄 Réinitialisation de l'agent en raison d'un manque d'amélioration.")
            agent.reset()
            no_improvement_count = 0

    episode_count += 1

assert np.mean(rewards_sarsa[-100:]) > 0.0

env.close() 
plt.plot(rewards_sarsa, color='cyan')
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.title('SARSA Agent Performance')
plt.show()

def plot_performance(rewards_q, rewards_q_eps, rewards_sarsa):
    plt.figure(figsize=(12, 8))

    # Calcul de la moyenne mobile pour lisser les courbes 
    def moving_average(rewards, window_size):
        return np.convolve(rewards, np.ones((window_size,))/window_size, mode='valid')

    episodes = 100
    smoothed_rewards_q = moving_average(rewards_q, episodes)
    smoothed_rewards_q_eps = moving_average(rewards_q_eps, episodes)
    smoothed_rewards_sarsa = moving_average(rewards_sarsa, episodes)

    plt.plot(smoothed_rewards_q, label='Q-Learning Agent', color='purple')
    plt.plot(smoothed_rewards_q_eps, label='Q-Learning Agent (Eps Scheduling)', color='orange')
    plt.plot(smoothed_rewards_sarsa, label='SARSA Agent', color='cyan')
    plt.xlabel('Episodes')
    plt.ylabel('Smoothed Total Reward')
    plt.title('Comparison of Agent Performances')
    plt.legend()
    plt.grid(True)
    plt.ylim(-50, 20)  # Adjusted scale for better visualization

    plt.show()

plot_performance(rewards_q, rewards_q_eps, rewards_sarsa)