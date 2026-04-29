import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import pygame
import random

from multiagent import FootballMultiAgentEnv
from policy import *
from pen_football import *

def play_zero_player(policy1: str | tuple, policy2: str, episodes=None):
    """
    Loads policies from a checkpoint file and evaluates them.
    """

    if isinstance(policy1, tuple):
        pname, kwargs = policy1
        policy_red = make_policy(pname, **kwargs)
    else:
        policy_red, _ = policy_from_checkpoint_path(policy1)

    if isinstance(policy2, tuple):
        pname, kwargs = policy2
        policy_blue = make_policy(pname, **kwargs)
    else:
        policy_blue, _ = policy_from_checkpoint_path(policy2)

    policy_red.eval()
    policy_blue.eval()

    env = FootballMultiAgentEnv({"render_mode": "human"})
    clock = pygame.time.Clock()
    scores = []
    ep = 0
    while True:
        obs, _ = env.reset(reset_score=False)
        done = False
        total_reward = np.zeros(4)

        while not done:
            with torch.no_grad():
                a_red = policy_red.sample_action(obs["player_red"])
                a_blue = policy_blue.sample_action(obs["player_blue"])

            obs, rewards, terminated, truncated, _ = env.step(
                {"player_red": a_red, "player_blue": a_blue}
            )
            done = terminated["__all__"] or truncated["__all__"]
            total_reward += np.array(rewards["player_red"])

            env.render()
            clock.tick(30)

        print(f"Episode {ep} final reward for red: {total_reward.sum():.1f} (Score: {total_reward[0]:.1f}, Move: {total_reward[1]:.1f}, Kick: {total_reward[2]:.1f}, Jump: {total_reward[3]:.1f})")
        scores.append(total_reward)

        ep += 1
        if episodes:
            if ep == episodes:
                break
        else:
            if env.game.score_red == 10 or env.game.score_blue == 10:
                break

    env.close()
    return scores

if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((BASE_WIDTH, BASE_HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("Pen Football - One Player")
    # play_zero_player(
    #     "../checkpoints/league_ppo_real (score reward)/checkpoint_6000000.pth",
    #     "../checkpoints/shoot_left_ppo (without embedding)/checkpoint_2998272.pth",
    #     )
    play_zero_player(
        # "../checkpoints/league_ppo_real_feudal/checkpoint_29500000.pth",
        "../checkpoints/league_ppo (misc rewards)/checkpoint_7100000.pth",
        # "../checkpoints/league_stable_curated_run1/checkpoint_54000000.pth",
        # "../checkpoints/league_stable_curated_run1/checkpoint_56000000.pth",
        "../checkpoints/league_stable_generalist_run2/checkpoint_65000000.pth"
        )
