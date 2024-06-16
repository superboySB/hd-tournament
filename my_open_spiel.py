from typing import Optional

import numpy as np
from gymnasium.spaces import Box, Discrete

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.env.utils import try_import_pyspiel

pyspiel = try_import_pyspiel(error=True)


class OpenSpielEnv(MultiAgentEnv):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self._skip_env_checking = True
        # Agent IDs are ints, starting from 0.
        self.num_agents = self.env.num_players()
        # Store the open-spiel game type.
        self.type = self.env.get_type()
        # Stores the current open-spiel game state.
        self.state = None

        # Extract observation- and action spaces from game.
        # (-inf, inf, 120)
        self.observation_space = Box(
            float("-inf"), float("inf"), (self.env.observation_tensor_size(),)
        )
        # Descrete(5)
        self.action_space = Discrete(self.env.num_distinct_actions())

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self.state = self.env.new_initial_state()
        return self._get_obs(), {}

    def step(self, action):
        # Before applying action(s), there could be chance nodes.
        # E.g. if env has to figure out, which agent's action should get
        # resolved first in a simultaneous node.
        self._solve_chance_nodes()
        penalties = {}

        # Sequential game:
        if str(self.type.dynamics) == "Dynamics.SEQUENTIAL":
            curr_player = self.state.current_player()
            assert curr_player in action
            try:
                self.state.apply_action(action[curr_player])
            # TODO: (sven) resolve this hack by publishing legal actions
            #  with each step.
            except pyspiel.SpielError:
                self.state.apply_action(np.random.choice(self.state.legal_actions()))
                penalties[curr_player] = -0.1

            # Compile rewards dict.
            rewards = {ag: r for ag, r in enumerate(self.state.returns())}
        # Simultaneous game.  Markov games
        else:
            assert self.state.current_player() == -2
            # Apparently, this works, even if one or more actions are invalid.
            self.state.apply_actions([action[ag] for ag in range(self.num_agents)])

        # Now that we have applied all actions, get the next obs.
        obs = self._get_obs()

        # Compile rewards dict and add the accumulated penalties
        # (for taking invalid actions).
        rewards = {ag: r for ag, r in enumerate(self.state.returns())}
        for ag, penalty in penalties.items():
            rewards[ag] += penalty

        # Are we done?
        is_terminated = self.state.is_terminal()
        terminateds = dict(
            {ag: is_terminated for ag in range(self.num_agents)},
            **{"__all__": is_terminated}
        )
        truncateds = dict(
            {ag: False for ag in range(self.num_agents)}, **{"__all__": False}
        )

        return obs, rewards, terminateds, truncateds, {}

    def render(self, mode=None) -> None:
        if mode == "human":
            print(self.state)

    def _get_obs(self):
        # Before calculating an observation, there could be chance nodes
        # (that may have an effect on the actual observations).
        # E.g. After reset, figure out initial (random) positions of the
        # agents.
        self._solve_chance_nodes()

        if self.state.is_terminal():
            return {}

        # Sequential game:
        if str(self.type.dynamics) == "Dynamics.SEQUENTIAL":
            curr_player = self.state.current_player()
            return {curr_player: np.reshape(self.state.observation_tensor(), [-1])}
        # Simultaneous game.
        else:
            assert self.state.current_player() == -2
            return {
                ag: np.reshape(self.state.observation_tensor(ag), [-1])
                for ag in range(self.num_agents)
            }

    def _solve_chance_nodes(self):
        # Chance node(s): Sample a (non-player) action and apply.
        while self.state.is_chance_node():
            assert self.state.current_player() == -1
            actions, probs = zip(*self.state.chance_outcomes())
            action = np.random.choice(actions, p=probs)
            self.state.apply_action(action)

if __name__ == "__main__":
    game = pyspiel.load_game("markov_soccer")
    env = OpenSpielEnv(game)  # 类似洪都的游戏内核
    obs = env.reset()
    done = {"__all__": False}
    step = 0
    while not done["__all__"]:
        actions = {agent: env.action_space.sample() for agent in range(env.num_agents)}  # {0: 3, 1: 1}
        obs, rewards, terminateds, truncateds, info = env.step(actions)  
        print(f"Step: {step}")
        print("Observations:", obs) # {0: np.array[120], 1: np.array[120]}
        print("Rewards:", rewards)  # {0: 1.0, 1: 1.0}
        done = terminateds  # {0: False, 1: fasle, __all__: false}, no info
        step += 1
        env.render(mode='human')
    print("Game over.")


