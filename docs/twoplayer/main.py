import pygame
import asyncio
import math

# Game Canvas
BASE_WIDTH, BASE_HEIGHT = 480, 360

# --- Game Constants (in base coordinates) ---
GROUND_Y = -151
CEILING_Y = 150
WALL_X = 230
TICK_RATE = 30

# --- Colors ---
COLOR_SKY = (204, 255, 255)
COLOR_GRASS = (0, 153, 51)
COLOR_BLACK = (0, 0, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_RED = (255, 0, 0)
COLOR_BLUE = (0, 51, 255)

class FootballGame:
    """
    A direct and faithful translation of the Scratch football game's core mechanics.
    """
    def __init__(self, screen=None, render_mode='human'):
        self.screen = screen
        self.clock = None
        self.scale = 2
        self.width = BASE_WIDTH * self.scale
        self.height = BASE_HEIGHT * self.scale
        self.render_mode = render_mode
        self.font = None
        self.reset()

    def get_font(self):
        """Gets font object - using custom font"""
        try:
            return pygame.font.Font("assets/JetBrainsMono-Regular.ttf", int(40 * self.scale))
        except:
            # Fallback to default if font not found
            return pygame.font.Font(None, int(40 * self.scale))

    def _s2p(self, x, y):
        """Converts Scratch coordinates to scaled Pygame screen coordinates."""
        return int((x * self.scale) + self.width / 2), int(self.height / 2 - (y * self.scale))

    def _reset_round(self):
        """Resets positions and velocities for a new round (e.g., after a goal)."""
        self.ball = {'x': 0, 'y': 0, 'vx': 0, 'vy': 0}
        self.red = {'x': -200, 'y': -130, 'vx': 0, 'vy': 0, 'can_double_jump': False, 'is_waiting_for_jump_key_release': False}
        self.blue = {'x': 200, 'y': -130, 'vx': 0, 'vy': 0, 'can_double_jump': False, 'is_waiting_for_jump_key_release': False}

    def reset(self):
        """Resets the entire game to its initial state for a new episode."""
        self._reset_round()
        self.score_red = 0
        self.score_blue = 0
        self.time_steps = 0

    def _update_player(self, player, keys):
        is_on_ground = player['y'] <= GROUND_Y

        if player['is_waiting_for_jump_key_release'] and not keys['jump']:
            player['can_double_jump'] = True
            player['is_waiting_for_jump_key_release'] = False

        if keys['jump']:
            if is_on_ground:
                player['vy'] = 12
                player['is_waiting_for_jump_key_release'] = True
            elif player['can_double_jump'] and player['vy'] < 5:
                player['vy'] = 12
                player['can_double_jump'] = False

        if keys['right']: player['vx'] += 1
        if keys['left']: player['vx'] -= 1

        player['vy'] -= 1
        player['x'] += player['vx']
        player['y'] += player['vy']
        player['vx'] *= 0.9

        if player['x'] > WALL_X: player['x'], player['vx'] = WALL_X, 0
        if player['x'] < -WALL_X: player['x'], player['vx'] = -WALL_X, 0
        if player['y'] > CEILING_Y: player['y'], player['vy'] = CEILING_Y, 0
        if player['y'] < GROUND_Y:
            player['y'], player['vy'] = GROUND_Y, 0
            player['can_double_jump'] = False
            player['is_waiting_for_jump_key_release'] = False

    def _update_ball(self):
        def process_collision(player):
            dx = self.ball['x'] - player['x']
            self.ball['vx'] = (player['vx'] * abs(dx)) / 5 + dx / 5
            self.ball['vy'] = player['vy'] + 10

        # Using math.hypot instead of np.hypot
        if math.hypot(self.ball['x'] - self.red['x'], self.ball['y'] - self.red['y']) < 20:
            process_collision(self.red)
        if math.hypot(self.ball['x'] - self.blue['x'], self.ball['y'] - self.blue['y']) < 20:
            process_collision(self.blue)

        self.ball['vy'] -= 1
        self.ball['vx'] *= 0.97
        self.ball['x'] += self.ball['vx']
        self.ball['y'] += self.ball['vy']

        # Using math.copysign instead of np.sign
        if abs(self.ball['x']) > WALL_X:
            self.ball['x'] = math.copysign(WALL_X, self.ball['x'])
            self.ball['vx'] = self.ball['vx'] * -0.7
        if self.ball['y'] < GROUND_Y:
            self.ball['y'], self.ball['vy'] = GROUND_Y, self.ball['vy'] * -0.7
        if self.ball['y'] > CEILING_Y:
            self.ball['y'], self.ball['vy'] = CEILING_Y, self.ball['vy'] * -0.7

        if abs(self.ball['x']) > 205 and self.ball['y'] > -40 and self.ball['y'] + self.ball['vy'] <= -40:
            self.ball['y'], self.ball['vy'] = -40, 5
            self.ball['vx'] = -5 * math.copysign(1, self.ball['x'])

    def render(self):
        if self.render_mode != 'human':
            return

        if self.screen is None:
            # First-time setup
            pygame.init()
            pygame.font.init()
            pygame.display.set_caption("Football RL")
            self.screen = pygame.display.set_mode((self.width, self.height))
            self.clock = pygame.time.Clock()

        self.screen.fill(COLOR_SKY)

        # Goals (drawn first)
        for side in [-1, 1]:
            color = COLOR_RED if side == -1 else COLOR_BLUE
            x_post = side * 230

            # --- Outline ---
            post_rect_outline_tl = self._s2p(x_post - 10, -50)
            pygame.draw.rect(self.screen, COLOR_BLACK, (post_rect_outline_tl[0], post_rect_outline_tl[1], int(20 * self.scale), int(self.scale * 120)))
            crossbar_x_start = x_post if side == 1 else x_post - 10
            crossbar_rect_outline_tl = self._s2p(crossbar_x_start, -40)
            pygame.draw.rect(self.screen, COLOR_BLACK, (crossbar_rect_outline_tl[0], crossbar_rect_outline_tl[1], int(10 * self.scale), int(20 * self.scale)))
            pygame.draw.circle(self.screen, COLOR_BLACK, self._s2p(x_post, -50), int(10 * self.scale))

            # --- Fill ---
            post_rect_fill_tl = self._s2p(x_post - 8, -50)
            pygame.draw.rect(self.screen, color, (post_rect_fill_tl[0], post_rect_fill_tl[1], int(16 * self.scale), int(120 * self.scale)))
            crossbar_x_fill_start = x_post if side == 1 else x_post - 8
            crossbar_rect_fill_tl = self._s2p(crossbar_x_fill_start, -42)
            pygame.draw.rect(self.screen, color, (crossbar_rect_fill_tl[0], crossbar_rect_fill_tl[1], int(8 * self.scale), int(16 * self.scale)))
            pygame.draw.circle(self.screen, color, self._s2p(x_post, -50), int(8 * self.scale))

        # Ground
        pygame.draw.line(self.screen, COLOR_BLACK, self._s2p(-240, -170), self._s2p(240, -170), int(20 * self.scale))
        pygame.draw.line(self.screen, COLOR_GRASS, self._s2p(-240, -170), self._s2p(240, -170), int(16 * self.scale))

        # Boundary
        pygame.draw.rect(self.screen, COLOR_BLACK, (0, 0, self.width, self.height), int(2 * self.scale))

        # Players and ball
        pygame.draw.circle(self.screen, COLOR_BLACK, self._s2p(self.red['x'], self.red['y']), int(10 * self.scale))
        pygame.draw.circle(self.screen, COLOR_RED, self._s2p(self.red['x'], self.red['y']), int(8 * self.scale))
        pygame.draw.circle(self.screen, COLOR_BLACK, self._s2p(self.blue['x'], self.blue['y']), int(10 * self.scale))
        pygame.draw.circle(self.screen, COLOR_BLUE, self._s2p(self.blue['x'], self.blue['y']), int(8 * self.scale))
        pygame.draw.circle(self.screen, COLOR_BLACK, self._s2p(self.ball['x'], self.ball['y']), int(10 * self.scale))
        pygame.draw.circle(self.screen, COLOR_WHITE, self._s2p(self.ball['x'], self.ball['y']), int(8 * self.scale))

        score_text = self.get_font().render(f"{self.score_red} - {self.score_blue}", True, COLOR_BLACK)
        self.screen.blit(score_text, (self.width/2 - score_text.get_width()/2, int(10 * self.scale)))

        pygame.display.flip()

    def close(self):
        """Shuts down the Pygame instance if it was created."""
        if self.screen is not None:
            pygame.quit()
            self.screen = None

# Initialize pygame and screen
pygame.init()
screen = pygame.display.set_mode((BASE_WIDTH * 2, BASE_HEIGHT * 2))
clock = pygame.time.Clock()

# Pygbag customization - changes the loading screen appearance
import sys
if sys.platform == "emscripten":
    import platform
    platform.window.document.title = "Pen Football - Two Player Game"

async def main():
    pygame.display.set_caption("Pen Football - Two Player")
    game = FootballGame(screen)
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        red_keys = {'jump': keys[pygame.K_w], 'left': keys[pygame.K_a], 'right': keys[pygame.K_d]}
        blue_keys = {'jump': keys[pygame.K_UP], 'left': keys[pygame.K_LEFT], 'right': keys[pygame.K_RIGHT]}

        game._update_player(game.red, red_keys)
        game._update_player(game.blue, blue_keys)
        game._update_ball()

        if game.ball['y'] < -50:
            if game.ball['x'] > 210:
                game.score_red += 1
                game._reset_round()
            elif game.ball['x'] < -210:
                game.score_blue += 1
                game._reset_round()

        if game.score_red >= 10 or game.score_blue >= 10:
            print(f"Game Over! Final Score: Red {game.score_red} - Blue {game.score_blue}")
            game.reset()

        game.render()
        clock.tick(TICK_RATE)

        # CRITICAL: This allows the browser to process events
        await asyncio.sleep(0)

    pygame.quit()

# Run the game
asyncio.run(main())
