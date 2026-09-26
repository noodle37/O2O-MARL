from typing import Any, Dict

import numpy as np
from envs.custom.multiagent_particle_envs.multiagent.environment import (
    MultiAgentEnv,
)
from envs.custom.multiagent_particle_envs.multiagent.simple_spread import Scenario

from .base import BaseEnvironment, ResetReturn, StepReturn


class MPEOMAR(BaseEnvironment):
    """Note: currently only supports simple spread."""

    def __init__(self, scenario, seed=None):
        # load scenario from script
        scenario = Scenario()  # TODO: make variable
        # create world
        world = scenario.make_world()

        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)

        self.environment = env

        self.agents = [f"agent_{n}" for n in range(3)]
        self.num_actions = 2
        self.num_agents = len(self.agents)
        self.observation_shape = None
        self.state_shape = None

        self.max_episode_length = 25

        self.t = 0
        self._init_shapes()

    def _init_shapes(self) -> None:
        obs = self.environment.reset()
        sample_obs = obs[0].astype("float32")
        self.observation_shape = (int(sample_obs.shape[0]),)
        state = np.concatenate([obs[i].astype("float32") for i in range(len(self.agents))])
        self.state_shape = (int(state.shape[0]),)

    def reset(self) -> ResetReturn:
        obs = self.environment.reset()

        observations = {agent: obs[i].astype("float32") for i, agent in enumerate(self.agents)}

        self.t = 0

        state = np.concatenate([obs[i].astype("float32") for i in range(len(self.agents))])
        self._set_shapes_from_obs(observations, state)
        return observations, {"state": state}

    def step(self, actions: Dict[str, np.ndarray]) -> StepReturn:
        mpe_actions = []
        for agent in self.agents:
            mpe_actions.append(actions[agent])

        next_observation, reward, done, info = self.environment.step(mpe_actions)

        terminals = {agent: done[i] for i, agent in enumerate(self.agents)}
        trunctations = {agent: False for i, agent in enumerate(self.agents)}

        rewards = {agent: reward[i] for i, agent in enumerate(self.agents)}

        observations = {
            agent: next_observation[i].astype("float32") for i, agent in enumerate(self.agents)
        }

        if self.t == 25:
            terminals = {agent: True for i, agent in enumerate(self.agents)}

        self.t += 1

        state = np.concatenate(
            [next_observation[i].astype("float32") for i in range(len(self.agents))]
        )
        info = {"state": state}

        return observations, rewards, terminals, trunctations, info  # type: ignore

    def _set_shapes_from_obs(
        self, observations: Dict[str, np.ndarray], state: np.ndarray
    ) -> None:
        sample_obs = next(iter(observations.values()))
        self.observation_shape = (int(sample_obs.shape[0]),)
        self.state_shape = (int(state.shape[0]),)

    def __getattr__(self, name: str) -> Any:
        """Expose any other attributes of the underlying environment."""
        if hasattr(self.__class__, name):
            return self.__getattribute__(name)
        else:
            return getattr(self.environment, name)
