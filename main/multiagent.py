import pygame
import numpy as np
import random
import time
import gymnasium as gym
from gymnasium import spaces

from pen_football import *

class FootballMultiAgentEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, config=None):
        super().__init__()
        render_mode = config.get("render_mode") if config else None
        self.game = FootballGame(render_mode=render_mode)

        self.possible_agents = ["player_red", "player_blue"]

        # --- Spaces ---
        # Each agent has 3 discrete actions (left/right/jump)
        single_action = spaces.Dict({
            "left": spaces.Discrete(2),
            "right": spaces.Discrete(2),
            "jump": spaces.Discrete(2)
        })

        self.action_space = spaces.Dict({
            agent: single_action for agent in self.possible_agents
        })

        # Observations: 12 floats per agent
        low = np.float32(-1.5)
        high = np.float32(1.5)

        self.observation_space = spaces.Dict({
            agent: spaces.Box(low=low, high=high, shape=(12,), dtype=np.float32)
            for agent in self.possible_agents
        })
        self.setting = None
        self.history = {}

    def set_setting(self, drill):
        """Sets setting"""
        self.setting = drill

    # -----------------------------------------------------

    def _get_obs(self):
        game_obs = self.game._get_internal_observation()
        red_state = game_obs[0:4]
        blue_state = game_obs[4:8]
        ball_state = game_obs[8:12]

        # mirror for blue
        flip = np.array([-1, 1, -1, 1], dtype=np.float32)
        blue_state_sym = blue_state * flip
        red_state_sym = red_state * flip
        ball_state_sym = ball_state * flip

        obs_red = np.concatenate([red_state, blue_state, ball_state]).astype(np.float32)
        obs_blue = np.concatenate([blue_state_sym, red_state_sym, ball_state_sym]).astype(np.float32)

        return {
            "player_red": obs_red,
            "player_blue": obs_blue
        }

    # -----------------------------------------------------

    def reset(self, *, seed=None, options=None, reset_score=True):
        self.history = {}
        if options and options.get("state") is not None:
            self.game.set_sim_state(
                options["state"],
                reset_score=options.get("reset_score", reset_score),
                reset_time_steps=options.get("reset_time_steps", True),
            )
        elif self.setting is None:
            self.game.reset(reset_score=reset_score)
        elif self.setting["drill"] == "block":
            data = np.load("../samples/block.npy")
            ind = random.randint(0, 999)
            ball_obs = data[ind, :]
            red_obs = np.array([random.uniform(-1, random.uniform(-1, ball_obs[0])), random.uniform(-1, random.uniform(-1, ball_obs[1])), random.gauss(0, 3/20), random.gauss(0, 3/20)])
            # red_obs = np.array([-200/230, -130/150, 0, 0])
            blue_obs = np.array([1, -1, 0, 0])
            obs = np.concatenate([red_obs, blue_obs, ball_obs])
            self.game.preset(obs, reset_score=reset_score)
        elif self.setting["drill"] == "block_nobounce":
            data = np.load("../samples/block_nobounce.npy")
            ind = random.randint(0, 999)
            ball_obs = data[ind, :]
            red_obs = np.array([random.uniform(-1, random.uniform(-1, ball_obs[0])), random.uniform(-1, random.uniform(-1, ball_obs[1])), random.gauss(0, 3/20), random.gauss(0, 3/20)])
            # red_obs = np.array([-200/230, -130/150, 0, 0])
            blue_obs = np.array([1, -1, 0, 0])
            obs = np.concatenate([red_obs, blue_obs, ball_obs])
            self.game.preset(obs, reset_score=reset_score)
        elif self.setting["drill"] == "shoot_left":
            ball_obs = np.array([random.uniform(-0.9, 0.9), np.clip(random.uniform(-1.1, 1), -1, 1), random.gauss(0, 3/20), random.gauss(0, 5/20)])
            red_obs = np.array([random.uniform(-1, ball_obs[0]), random.uniform(-1, random.uniform(-1, ball_obs[1])), random.gauss(0, 3/20), random.gauss(0, 3/20)])
            blue_obs = np.array([1, -1, 0, 0])
            obs = np.concatenate([red_obs, blue_obs, ball_obs])
            self.game.preset(obs, reset_score=reset_score)
            self.game.draw("cross", 229/230, self.setting["par"])
        elif self.setting["drill"] == "shoot_right":
            ball_obs = np.array([random.uniform(-0.9, 0.9), np.clip(random.uniform(-1.1, 1), -1, 1), random.gauss(0, 3/20), random.gauss(0, 5/20)])
            red_obs = np.array([random.uniform(ball_obs[0], 1), random.uniform(-1, random.uniform(-1, ball_obs[1])), random.gauss(0, 3/20), random.gauss(0, 3/20)])
            blue_obs = np.array([1, -1, 0, 0])
            obs = np.concatenate([red_obs, blue_obs, ball_obs])
            self.game.preset(obs, reset_score=reset_score)
            self.game.draw("cross", 229/230, self.setting["par"])
        elif self.setting["drill"] == "hit_left_wall":
            ball_obs = np.array([random.uniform(-0.9, 0.9), np.clip(random.uniform(-1.1, 1), -1, 1), random.gauss(0, 3/20), random.gauss(0, 5/20)])
            red_obs = np.array([random.uniform(-1, 1), random.uniform(-1, random.uniform(-1, ball_obs[1])), random.gauss(0, 3/20), random.gauss(0, 3/20)])
            blue_obs = np.array([1, -1, 0, 0])
            obs = np.concatenate([red_obs, blue_obs, ball_obs])
            self.game.preset(obs, reset_score=reset_score)
            self.game.draw("cross", -229/230, self.setting["par"])
        elif self.setting["drill"] == "hit_right_wall":
            ball_obs = np.array([random.uniform(-0.9, 0.9), np.clip(random.uniform(-1.1, 1), -1, 1), random.gauss(0, 3/20), random.gauss(0, 5/20)])
            red_obs = np.array([random.uniform(-1, 1), random.uniform(-1, random.uniform(-1, ball_obs[1])), random.gauss(0, 3/20), random.gauss(0, 3/20)])
            blue_obs = np.array([1, -1, 0, 0])
            obs = np.concatenate([red_obs, blue_obs, ball_obs])
            self.game.preset(obs, reset_score=reset_score)
            self.game.draw("cross", 229/230, self.setting["par"])
        elif self.setting["drill"] == "volley":
            ball_obs = np.array([random.uniform(-0.9, 0.9), np.clip(random.uniform(-1.1, 1), -1, 1), random.gauss(0, 3/20), random.gauss(0, 5/20)])
            red_obs = np.array([random.uniform(-1, 1), random.uniform(-1, random.uniform(-1, ball_obs[1])), random.gauss(0, 3/20), random.gauss(0, 3/20)])
            blue_obs = np.array([1, -1, 0, 0])
            obs = np.concatenate([red_obs, blue_obs, ball_obs])
            self.game.preset(obs, reset_score=reset_score)
            self.game.draw("cross", ball_obs[0].item(), self.setting["par"])
        elif self.setting["drill"] == "prepare":
            ball_obs = np.array([random.uniform(-0.9, 0.9), np.clip(random.uniform(-1.1, 1), -1, 1), random.gauss(0, 3/20), random.gauss(0, 5/20)])
            red_obs = np.array([random.uniform(-1, 1), random.uniform(-1, random.uniform(-1, ball_obs[1])), random.gauss(0, 3/20), random.gauss(0, 3/20)])
            blue_obs = np.array([1, -1, 0, 0])
            obs = np.concatenate([red_obs, blue_obs, ball_obs])
            self.game.preset(obs, reset_score=reset_score)
            self.game.draw("cross", self.setting["par"], -1)
        else:
            self.game.reset(reset_score=reset_score)
        return self._get_obs(), {}

    # -----------------------------------------------------

    def step(self, action_dict):
        a_red = action_dict["player_red"]
        a_blue = action_dict["player_blue"]

        red_keys = {
            "left": a_red["left"] == 1,
            "right": a_red["right"] == 1,
            "jump": a_red["jump"] == 1
        }
        blue_keys = {
            "left": a_blue["right"] == 1,
            "right": a_blue["left"] == 1,
            "jump": a_blue["jump"] == 1
        }

        _, (red_state, blue_state, game_state), _, _, _ = self.game.step(red_keys, blue_keys)

        obs = self._get_obs()

        reports = self.comp_reports(red_state, blue_state)
        rewards = self.comp_rewards(red_state, blue_state, game_state)

        terminateds = self.comp_terminateds(red_state, blue_state, game_state)
        truncateds = self.comp_truncateds(game_state)

        if red_state['kicked']:
            self.history['red_kicked'] = True
        self.history['min_red_ball_dist'] = min(self.history.get('min_red_ball_dist', float('inf')), red_state['ball_dist'])

        result = None
        if terminateds["__all__"] or truncateds["__all__"]:
            if red_state["scored"]:
                result = "red"
            elif blue_state["scored"]:
                result = "blue"
            else:
                result = "draw"

        return obs, rewards, terminateds, truncateds, {"reports": reports, "result": result}

    # -----------------------------------------------------

    def render(self):
        self.game.render()

    def close(self):
        self.game.close()

    def comp_reports(self, red_state: dict, blue_state: dict):
        score_red = (red_state['scored'] - blue_state['scored']) * 1
        score_blue = (blue_state['scored'] - red_state['scored']) * 1

        move_red = (red_state['move_towards_ball']) * 1
        move_blue = (blue_state['move_towards_ball']) * 1

        kick_red = red_state['kicked'] * 1
        kick_blue = blue_state['kicked'] * 1

        jump_red = red_state['jump_failed'] * 1
        jump_blue = blue_state['jump_failed'] * 1

        dist_red = red_state['ball_dist'] * 1
        dist_blue = blue_state['ball_dist'] * 1

        reports = {
            "player_red": [float(score_red), float(move_red), float(kick_red), float(jump_red), float(dist_red)],
            "player_blue": [float(score_blue), float(move_blue), float(kick_blue), float(jump_blue), float(dist_blue)]
        }
        return reports

    def comp_rewards(self, red_state: dict, blue_state: dict, game_state: dict):
        if self.setting is None:
            score_red = (red_state['scored'] - blue_state['scored']) * 100
            score_blue = (blue_state['scored'] - red_state['scored']) * 100

            move_red = (red_state['move_towards_ball']) * 0.1
            move_blue = (blue_state['move_towards_ball']) * 0.1

            kick_red = red_state['kicked'] * 10
            kick_blue = blue_state['kicked'] * 10

            jump_red = red_state['jump_failed'] * (-1)
            jump_blue = blue_state['jump_failed'] * (-1)

            rewards = {
                # "player_red": score_red + move_red + kick_red + jump_red,
                "player_red": score_red,
                # "player_blue": score_blue + move_blue + kick_blue + jump_blue
                "player_blue": score_blue,
            }
        elif self.setting["drill"] == "block":
            score_red = blue_state["scored"] * (-10) + (game_state["time_steps"] == 90) * 10
            score_blue = 0

            rewards = {
                "player_red": score_red,
                "player_blue": score_blue
            }
        elif self.setting["drill"] == "block_nobounce":
            score_red = blue_state["scored"] * (-10) + (game_state["time_steps"] == 90) * 10
            score_blue = 0

            rewards = {
                "player_red": score_red,
                "player_blue": score_blue
            }
        elif self.setting["drill"] == "shoot_left":
            score_red = (red_state["scored"] * (40) * (2 - abs(game_state['ball_y'] - self.setting["par"]))
                        + (red_state["kicked"] * (1 - self.history.get("red_kicked", 0)) * 10)
                        + ((1 - self.history.get("red_kicked", 0)) * (min(self.history.get('min_red_ball_dist', float('inf')), red_state['ball_dist']) - 20) / 400 * (-0.1))
                        + ((1 - red_state["scored"]) * (-0.1))
                        + (game_state["time_steps"] == 180) * (-10))
            # score_red = red_state["scored"] * (40) * (2 - abs(game_state['ball_y'] - self.setting["par"]))
            score_blue = 0

            rewards = {
                "player_red": score_red,
                "player_blue": score_blue
            }
        elif self.setting["drill"] == "shoot_right":
            score_red = red_state["scored"] * (40) * (2 - abs(game_state['ball_y'] - self.setting["par"]))
            score_blue = 0

            rewards = {
                "player_red": score_red,
                "player_blue": score_blue
            }
        elif self.setting["drill"] == "hit_left_wall":
            score_red = game_state["ball_hit_left_wall"] * (40) * (2 - abs(game_state['ball_y'] - self.setting["par"]))
            score_blue = 0

            rewards = {
                "player_red": score_red,
                "player_blue": score_blue
            }
        elif self.setting["drill"] == "hit_right_wall":
            score_red = game_state["ball_hit_right_wall"] * (40) * (2 - abs(game_state['ball_y'] - self.setting["par"]))
            score_blue = 0

            rewards = {
                "player_red": score_red,
                "player_blue": score_blue
            }
        elif self.setting["drill"] == "volley":
            score_red = red_state["kicked"] * (5) * (10 - 10*abs(game_state['ball_y'] - self.setting["par"]))
            score_blue = 0

            rewards = {
                "player_red": score_red,
                "player_blue": score_blue
            }
        elif self.setting["drill"] == "prepare":
            score_red = (-1) * (abs(red_state['x'] - self.setting["par"])) + (-1) * (abs(red_state['y'] - 1)) + 0.01
            score_blue = 0

            rewards = {
                "player_red": score_red,
                "player_blue": score_blue
            }
        return rewards

    def comp_terminateds(self, red_state: dict, blue_state: dict, game_state: dict):
        if self.setting is None:
            terminateds = {"__all__": red_state['scored'] or blue_state['scored']}
        elif self.setting["drill"] == "block":
            terminateds = {"__all__": red_state['scored'] or blue_state['scored']}
        elif self.setting["drill"] == "block_nobounce":
            terminateds = {"__all__": red_state['scored'] or blue_state['scored']}
        elif self.setting["drill"] == "shoot_left":
            terminateds = {"__all__": red_state['scored'] or blue_state['scored']}
        elif self.setting["drill"] == "shoot_right":
            terminateds = {"__all__": red_state['scored'] or blue_state['scored']}
        elif self.setting["drill"] == "hit_left_wall":
            terminateds = {"__all__": game_state['ball_hit_left_wall']}
        elif self.setting["drill"] == "hit_right_wall":
            terminateds = {"__all__": game_state['ball_hit_right_wall']}
        elif self.setting["drill"] == "volley":
            terminateds = {"__all__": False}
        elif self.setting["drill"] == "prepare":
            terminateds = {"__all__": False}
        return terminateds

    def comp_truncateds(self, game_state: dict):
        if self.setting is None:
            truncateds = {"__all__": game_state['time_steps'] >= 1800}
        elif self.setting["drill"] == "block":
            truncateds = {"__all__": game_state['time_steps'] >= 90}
        elif self.setting["drill"] == "block_nobounce":
            truncateds = {"__all__": game_state['time_steps'] >= 90}
        elif self.setting["drill"] == "shoot_left":
            truncateds = {"__all__": game_state['time_steps'] >= 120}
        elif self.setting["drill"] == "shoot_right":
            truncateds = {"__all__": game_state['time_steps'] >= 120}
        elif self.setting["drill"] == "hit_left_wall":
            truncateds = {"__all__": game_state['time_steps'] >= 120}
        elif self.setting["drill"] == "hit_right_wall":
            truncateds = {"__all__": game_state['time_steps'] >= 120}
        elif self.setting["drill"] == "volley":
            truncateds = {"__all__": game_state['time_steps'] >= 90}
        elif self.setting["drill"] == "prepare":
            truncateds = {"__all__": game_state['time_steps'] >= 60}
        return truncateds
