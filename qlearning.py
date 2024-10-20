from collections import defaultdict
import random
import typing as t
import numpy as np
import gymnasium as gym

import math
Action = int
State = int
Info = t.TypedDict("Info", {"prob": float, "action_mask": np.ndarray})
QValues = t.DefaultDict[int, t.DefaultDict[Action, float]]


class QLearningAgent:
    def __init__(
        self,
        learning_rate: float,
        epsilon: float,
        gamma: float,
        legal_actions: t.List[Action],
    ):
        """
        Q-Learning Agent

        You should not use directly self._qvalues, but instead of its getter/setter.
        """
        self.legal_actions = legal_actions
        self._qvalues: QValues = defaultdict(lambda: defaultdict(int))
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma

    def get_qvalue(self, state: State, action: Action) -> float:
        """
        Returns Q(state,action)
        """
        return self._qvalues[state][action]

    def set_qvalue(self, state: State, action: Action, value: float):
        """
        Sets the Qvalue for [state,action] to the given value
        """
        self._qvalues[state][action] = value

    def get_value(self, state: State) -> float:
        """
        Compute your agent's estimate of V(s) using current q-values
        V(s) = max_a Q(s, a) over possible actions.
        """
        value = 0.0
        # BEGIN SOLUTION
        if not self._qvalues[state]:
            return 0.0
        actions_iter = iter(self._qvalues[state].values())
        max_value_found = False

        while True:
            try:
                current_value = next(actions_iter)
                if not max_value_found or current_value > value:
                    value = current_value
                    max_value_found = True
            except StopIteration:
                break
        # END SOLUTION
        return value

    def update(
        self, state: State, action: Action, reward: t.SupportsFloat, next_state: State
    ):
        """
        You should do your Q-Value update here (s'=next_state):
           TD_target(s') = r + gamma * max_a' Q(s', a')
           TD_error(s', a) = TD_target(s') - Q_old(s, a)
           Q_new(s, a) := Q_old(s, a) + learning_rate * TD_error(s', a)
        """
        q_value = 0.0
        # BEGIN SOLUTION
        current_q = self.get_qvalue(state, action)
        next_max_q = self.get_value(next_state)

        # Normaliser la récompense
        normalized_reward = reward / 20.0  # Supposons que la récompense maximale est 20
        target = normalized_reward + self.gamma * next_max_q
        td_error = target - current_q
        td_error = max(min(td_error, 1), -1)
        q_value = current_q + self.learning_rate * td_error

        self.set_qvalue(state, action, q_value)
        # END SOLUTION

    def get_best_action(self, state: State) -> Action:
        """
        Compute the best action to take in a state (using current q-values).
        """
        possible_q_values = []
        index = 0

        while index < len(self.legal_actions):
            possible_q_values.append(self.get_qvalue(state, self.legal_actions[index]))
            index += 1

        best_action_index = np.argmax(possible_q_values)
        return self.legal_actions[best_action_index]

    def get_action(self, state: State) -> Action:
        """
        Compute the action to take in the current state, including exploration.

        Exploration is done with epsilon-greedy. Namely, with probability self.epsilon, we should take a random action, and otherwise the best policy action (self.get_best_action).

        Note: To pick randomly from a list, use random.choice(list).
              To pick True or False with a given probability, generate a uniform number in [0, 1]
              and compare it with your probability
        """
        action = self.legal_actions[0]

        # BEGIN SOLUTION
        if not hasattr(self, 'epsilon'):
            self.epsilon = 1.0
        if not hasattr(self, 'update_count'):
            self.update_count = 0

        # Décroissance exponentielle de epsilon
        self.epsilon = max(0.01, min(1.0, 1.0 - math.log10((self.update_count + 1) / 25)))
        self.update_count += 1

        random_value = random.random()
        if random_value < self.epsilon:
            # Exploration: choisir une action aléatoire, mais favoriser les actions moins explorées
            action_counts = []
            index = 0

            while index < len(self.legal_actions):
                action_counts.append(self.get_qvalue(state, self.legal_actions[index]))
                index += 1

            min_count = min(action_counts)
            actions = []

            index = 0
            while index < len(self.legal_actions):
                if action_counts[index] == min_count:
                    actions.append(self.legal_actions[index])
                index += 1

            action = random.choice(actions)
        else:
            # Exploitation: choisir la meilleure action
            action = self.get_best_action(state)
        # END SOLUTION

        return action

    def reset(self):
        # BEGIN SOLUTION
        self._qvalues = defaultdict(lambda: defaultdict(int))
        self.update_count = 0
        self.epsilon = 1.0
        # END SOLUTION
