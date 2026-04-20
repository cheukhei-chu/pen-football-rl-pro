import asyncio
import pygame
import json
import math
import random
import sys
from urllib.parse import parse_qs

# Import the game - we'll need a numpy-free version for Pygbag
try:
    from pen_football_web import FootballGame, BASE_WIDTH, BASE_HEIGHT, TICK_RATE
except:
    # If import fails, the user needs pen_football.py in the same folder
    print("ERROR: pen_football_web.py not found!")
    raise

# --- PURE PYTHON NEURAL NETWORK (NO NUMPY) ---
W = {}
BOT_MANIFEST = {"default_bot": None, "bots": []}
CURRENT_BOT = {"name": "Default", "rating": None}

def load_bot_manifest():
    global BOT_MANIFEST
    try:
        with open("assets/bots_manifest.json", "r") as f:
            BOT_MANIFEST = json.load(f)
    except:
        BOT_MANIFEST = {"default_bot": None, "bots": []}

def get_selected_bot_id():
    if sys.platform != "emscripten":
        return None
    try:
        import platform
        search = str(platform.window.location.search or "")
        params = parse_qs(search.lstrip("?"))
        values = params.get("bot", [])
        return values[0] if values else None
    except:
        return None

def get_bot_by_id(bot_id):
    for bot in BOT_MANIFEST.get("bots", []):
        if bot["id"] == bot_id:
            return bot
    return None

def load_weights():
    global W, CURRENT_BOT
    load_bot_manifest()
    selected_bot_id = get_selected_bot_id() or BOT_MANIFEST.get("default_bot")
    bot = get_bot_by_id(selected_bot_id) if selected_bot_id else None
    weights_path = "assets/model_weights.json"
    if bot:
        weights_path = bot.get("weights_path", weights_path)
        CURRENT_BOT = {
            "name": bot.get("name", "Bot"),
            "rating": bot.get("rating"),
        }
    else:
        CURRENT_BOT = {"name": "Default", "rating": None}

    try:
        with open(weights_path, "r") as f:
            W = json.load(f)
        if bot and CURRENT_BOT["rating"] is not None:
            print(f"Weights loaded for {CURRENT_BOT['name']} (BT {CURRENT_BOT['rating']:.2f}).")
        else:
            print("Weights loaded successfully.")
    except:
        print(f"WARNING: Could not load {weights_path}! AI will play randomly.")
        CURRENT_BOT = {"name": "Random", "rating": None}

def dot_product(vec, mat):
    """Matrix multiply: vec (N,) * mat^T (MxN) = result (M,)
    mat is stored as (M, N) so we compute: result[j] = sum(vec[i] * mat[j][i])
    """
    result = []
    for j in range(len(mat)):  # Iterate over rows of mat
        total = sum(vec[i] * mat[j][i] for i in range(len(vec)))
        result.append(total)
    return result

def add_bias(vec, bias):
    return [vec[i] + bias[i] for i in range(len(vec))]

def relu(vec):
    return [max(0, x) for x in vec]

def linear(x, name):
    if f"{name}_w" not in W:
        return [0]
    w = W[f"{name}_w"]
    b = W[f"{name}_b"]
    result = dot_product(x, w)
    return add_bias(result, b)

def softmax_sample(logits):
    max_logit = max(logits)
    exp_logits = [math.exp(l - max_logit) for l in logits]
    total = sum(exp_logits)
    probs = [e / total for e in exp_logits]
    return 1 if random.random() < probs[1] else 0

def get_action(obs):
    if not W:
        return {"left": 0, "right": 0, "jump": 0}

    # Plan Net
    p = linear(obs, "plan_0")
    p = relu(p)
    p = linear(p, "plan_2")
    p = relu(p)

    # Action Net
    obs_subset = [obs[0], obs[1], obs[2], obs[3], obs[8], obs[9], obs[10], obs[11]]
    x = obs_subset + p
    x = linear(x, "action_0")
    x = relu(x)
    x = linear(x, "action_2")
    x = relu(x)

    # Heads
    left_logits = linear(x, "head_left")
    right_logits = linear(x, "head_right")
    jump_logits = linear(x, "head_jump")

    return {
        "left": softmax_sample(left_logits),
        "right": softmax_sample(right_logits),
        "jump": softmax_sample(jump_logits)
    }

def get_obs_for_blue(game):
    """Convert game observation to list (no numpy) and flip for blue player"""
    game_obs = game._get_internal_observation()

    # Convert numpy array to list if needed
    if hasattr(game_obs, 'tolist'):
        game_obs = game_obs.tolist()

    flip = [-1, 1, -1, 1]
    obs_blue = []

    # Blue player (flipped)
    for i in range(4):
        obs_blue.append(game_obs[4 + i] * flip[i])
    # Red player (flipped)
    for i in range(4):
        obs_blue.append(game_obs[i] * flip[i])
    # Ball (flipped positions, not velocities)
    obs_blue.append(game_obs[8] * -1)  # ball x
    obs_blue.append(game_obs[9])        # ball y
    obs_blue.append(game_obs[10] * -1)  # ball vx
    obs_blue.append(game_obs[11])       # ball vy

    return obs_blue

# Initialize pygame and screen
pygame.init()
SCALE = 2
screen = pygame.display.set_mode((BASE_WIDTH * SCALE, BASE_HEIGHT * SCALE))
clock = pygame.time.Clock()

async def main():
    pygame.display.set_caption("Pen Football - Single Player")
    game = FootballGame(screen)
    game.scale = SCALE
    game.width = BASE_WIDTH * SCALE
    game.height = BASE_HEIGHT * SCALE

    load_weights()
    info_font = pygame.font.SysFont("consolas", 14 * SCALE)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        red_keys = {
            'jump': keys[pygame.K_w] or keys[pygame.K_UP],
            'left': keys[pygame.K_a] or keys[pygame.K_LEFT],
            'right': keys[pygame.K_d] or keys[pygame.K_RIGHT]
        }

        obs_blue = get_obs_for_blue(game)
        a_blue = get_action(obs_blue)

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
        rating_suffix = ""
        if CURRENT_BOT["rating"] is not None:
            rating_suffix = f"  |  BT {CURRENT_BOT['rating']:.0f}"
        label = info_font.render(f"Opponent: {CURRENT_BOT['name']}{rating_suffix}", True, (20, 20, 20))
        screen.blit(label, (10, game.height - label.get_height() - 8))
        pygame.display.flip()
        clock.tick(TICK_RATE)

        await asyncio.sleep(0)

    pygame.quit()

asyncio.run(main())
