# Copyright 2023 InstaDeep Ltd. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Dict, Optional

import numpy as np
try:
    from multiagent_mujoco.mujoco_multi import MujocoMulti
except ImportError:
    from envs.custom.multiagent_mujoco.mujoco_multi import MujocoMulti

from envs.base import BaseEnvironment, ResetReturn, StepReturn
from envs.discrete_mujoco_actions import (
    all_legal_mask,
    continuous_to_index,
    index_to_continuous,
    n_discrete_actions,
)


def get_mamujoco_args(scenario: str) -> Dict[str, Any]:
    env_args = {
        "agent_obsk": 1,
        "episode_limit": 1000,
        # Required for og_marl mamujoco vaults (2halfcheetah Good obs is 13-d).
        "global_categories": "qvel,qpos",
    }
    if scenario.lower() == "4ant":
        env_args["scenario"] = "Ant-v2"
        env_args["agent_conf"] = "4x2"
    elif scenario.lower() == "2ant":
        env_args["scenario"] = "Ant-v2"
        env_args["agent_conf"] = "2x4"
    elif scenario.lower() == "2halfcheetah":
        env_args["scenario"] = "HalfCheetah-v2"
        env_args["agent_conf"] = "2x3"
    else:
        raise ValueError("Not a valid mamujoco scenario")
    return env_args


class MAMuJoCo(BaseEnvironment):

    """Environment wrapper Multi-Agent MuJoCo."""

    def __init__(self, scenario: str, seed: Optional[int] = 42):
        env_args = get_mamujoco_args(scenario)
        self._environment = MujocoMulti(env_args=env_args, seed=seed)

        self.agents = [f"agent_{n}" for n in range(self._environment.n_agents)]
        self.num_actions = self._environment.n_actions
        self.observation_shape = (self._environment.get_obs_size(),)
        self.state_shape = (self._environment.get_state_size(),)

    def reset(self) -> ResetReturn:
        self._environment.reset()

        observations = self._environment.get_obs()

        observations = {
            agent: observations[i].astype("float32") for i, agent in enumerate(self.agents)
        }

        info = {"state": self._environment.get_state()}

        return observations, info

    def step(self, actions: Dict[str, np.ndarray]) -> StepReturn:
        mujoco_actions = []
        for agent in self.agents:
            mujoco_actions.append(actions[agent])

        reward, done, info = self._environment.step(mujoco_actions)

        terminals = {agent: np.array(done) for agent in self.agents}
        truncations = {agent: np.array(False) for agent in self.agents}

        rewards = {agent: reward for agent in self.agents}

        observations = self._environment.get_obs()

        observations = {
            agent: observations[i].astype("float32") for i, agent in enumerate(self.agents)
        }

        info["state"] = self._environment.get_state()

        return observations, rewards, terminals, truncations, info  # type: ignore

    def __getattr__(self, name: str) -> Any:
        """Expose any other attributes of the underlying environment."""
        if hasattr(self.__class__, name):
            return self.__getattribute__(name)
        else:
            return getattr(self._environment, name)


class DiscreteMAMuJoCo(MAMuJoCo):
    """MuJoCo with per-agent discrete torque bins for QMIX / OVMSE."""

    def __init__(self, scenario: str, seed: Optional[int] = 42):
        super().__init__(scenario, seed=seed)
        self._scenario = scenario
        self.num_actions = n_discrete_actions(scenario)
        self._legal = all_legal_mask(scenario)

    def _legals_dict(self) -> Dict[str, np.ndarray]:
        return {agent: self._legal.copy() for agent in self.agents}

    def reset(self) -> ResetReturn:
        observations, info = super().reset()
        info["legals"] = self._legals_dict()
        return observations, info

    def step(self, actions: Dict[str, np.ndarray]) -> StepReturn:
        mujoco_actions: list = []
        for agent in self.agents:
            a = actions[agent]
            if np.isscalar(a) or (isinstance(a, np.ndarray) and a.shape == ()):
                idx = int(a)
            else:
                idx = int(np.asarray(a).reshape(-1)[0])
            mujoco_actions.append(index_to_continuous(idx, self._scenario))
        continuous = {agent: mujoco_actions[i] for i, agent in enumerate(self.agents)}
        obs, rewards, terminals, truncations, info = super().step(continuous)
        info["legals"] = self._legals_dict()
        return obs, rewards, terminals, truncations, info
