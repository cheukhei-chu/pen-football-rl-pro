import os
import numpy as np
import pygame
import random

from pen_football import *
from policy import *

def play_one_player(policy: FootballPolicy | str):
    """
    Loads policy from a checkpoint file and let the player play it.
    """

    if not isinstance(policy, FootballPolicy):
        policy_blue, _ = policy_from_checkpoint_path(policy)
    else:
        policy_blue = policy

    policy_blue.eval()

    pygame.init()
    screen = pygame.display.set_mode((BASE_WIDTH, BASE_HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("Pen Football - One Player")
    clock = pygame.time.Clock()
    game = FootballGame(screen)
    # policy_blue.set_setting({"drill": "shoot_left", "par": random.uniform(-1, -40/150)})
    # policy_blue.set_setting({"drill": "block_nobounce"})

    def _get_obs():
        game_obs = game._get_internal_observation()
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

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.VIDEORESIZE:
                new_scale = event.w / BASE_WIDTH
                new_width = BASE_WIDTH * new_scale
                new_height = BASE_HEIGHT * new_scale
                screen = pygame.display.set_mode((new_width, new_height), pygame.RESIZABLE)
                game.scale, game.width, game.height = new_scale, new_width, new_height
        keys = pygame.key.get_pressed()
        red_keys = {
            'jump': keys[pygame.K_w] or keys[pygame.K_UP],
            'left': keys[pygame.K_a] or keys[pygame.K_LEFT],
            'right': keys[pygame.K_d] or keys[pygame.K_RIGHT]
            }
        a_blue = policy_blue.sample_action(_get_obs()["player_blue"])
        blue_keys = {
            "left": a_blue["right"] == 1,
            "right": a_blue["left"] == 1,
            "jump": a_blue["jump"] == 1
        }

        _, _, terminated, _, _ = game.step(red_keys, blue_keys)
        if terminated:
            print(f"Game Over! Final Score: Red {game.score_red} - Blue {game.score_blue}")
            game.reset()
        game.render()
        clock.tick(TICK_RATE)
    pygame.quit()


if __name__ == "__main__":
    # play_one_player("../checkpoints/league_ppo (misc rewards)/checkpoint_800000.pth")
    # play_one_player("../checkpoints/red_league_test2/football_episode_38000.pth")
    # play_one_player("../checkpoints/league_ppo (misc rewards)/checkpoint_7100000.pth",)
    # play_one_player("../checkpoints/league_ppo_regular_real (score reward) (latent_dims 128 128)/checkpoint_12000000.pth")
    # play_one_player("../checkpoints/league_ppo_real_warmstart_experts/checkpoint_44000000.pth")
    # play_one_player("../checkpoints/league_stable_seeded/checkpoint_29000000.pth")
    # play_one_player("../checkpoints/league_stable_archive_run1/checkpoint_4000000.pth")
    # play_one_player("../checkpoints/league_stable_curated_run2/checkpoint_47000000.pth")
    # play_one_player("../checkpoints/league_stable_generalist_run1/checkpoint_22000000.pth")
    # play_one_player("../checkpoints/league_stable_generalist_run2/checkpoint_65000000.pth")
    # play_one_player("../checkpoints/league_stable_generalist_run3_resume_2w/checkpoint_80000000.pth")
    # play_one_player("../checkpoints/league_stable_generalist_run4/checkpoint_119000000.pth")
    play_one_player("../checkpoints/league_stable_generalist_run5/checkpoint_138000000.pth")
