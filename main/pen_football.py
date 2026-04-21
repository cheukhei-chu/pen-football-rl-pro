import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces

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
        self.scale = 1
        self.width = BASE_WIDTH * self.scale
        self.height = BASE_HEIGHT * self.scale
        self.render_mode = render_mode
        self.font = None
        self.to_draw = []
        self.reset()

    def get_font(self):
        """Gets font object"""
        return pygame.font.SysFont("consolas", int(40 * self.scale))

    def _s2p(self, x, y):
        """Converts Scratch coordinates to scaled Pygame screen coordinates."""
        return int((x * self.scale) + self.width / 2), int(self.height / 2 - (y * self.scale))

    def _reset_round(self):
        """Resets positions and velocities for a new round (e.g., after a goal)."""
        self.red = {'x': -200, 'y': -130, 'vx': 0, 'vy': 0, 'can_double_jump': False, 'is_waiting_for_jump_key_release': False}
        self.blue = {'x': 200, 'y': -130, 'vx': 0, 'vy': 0, 'can_double_jump': False, 'is_waiting_for_jump_key_release': False}
        self.ball = {'x': 0, 'y': 0, 'vx': 0, 'vy': 0}

    def _get_internal_observation(self):
        return np.array([
            self.red['x']/WALL_X, self.red['y']/CEILING_Y, self.red['vx']/20, self.red['vy']/20,
            self.blue['x']/WALL_X, self.blue['y']/CEILING_Y, self.blue['vx']/20, self.blue['vy']/20,
            self.ball['x']/WALL_X, self.ball['y']/CEILING_Y, self.ball['vx']/20, self.ball['vy']/20,
        ], dtype=np.float32)

    def reset(self, reset_score=True):
        """Resets the entire game to its initial state for a new episode."""
        self._reset_round()
        if reset_score:
            self.score_red = 0
            self.score_blue = 0
        self.time_steps = 0
        self.to_draw = []
        return self._get_internal_observation()

    def get_sim_state(self):
        """Returns a full simulator snapshot that can be restored later."""
        return {
            'red': dict(self.red),
            'blue': dict(self.blue),
            'ball': dict(self.ball),
            'score_red': int(getattr(self, 'score_red', 0)),
            'score_blue': int(getattr(self, 'score_blue', 0)),
            'time_steps': int(getattr(self, 'time_steps', 0)),
        }

    def set_sim_state(self, state, reset_score=True, reset_time_steps=True):
        """Restores a previously captured simulator snapshot."""
        self.red = dict(state['red'])
        self.blue = dict(state['blue'])
        self.ball = dict(state['ball'])
        if reset_score:
            self.score_red = 0
            self.score_blue = 0
        else:
            self.score_red = int(state.get('score_red', 0))
            self.score_blue = int(state.get('score_blue', 0))
        self.time_steps = 0 if reset_time_steps else int(state.get('time_steps', 0))
        self.to_draw = []
        return self._get_internal_observation()

    def preset(self, obs, reset_score=True):
        """Resets the entire game to the give state for a new episode."""
        self.red = {'x': obs[0]*WALL_X, 'y': obs[1]*CEILING_Y, 'vx': obs[2]*20, 'vy': obs[3]*20, 'can_double_jump': False, 'is_waiting_for_jump_key_release': False}
        self.blue = {'x': obs[4]*WALL_X, 'y': obs[5]*CEILING_Y, 'vx': obs[6]*20, 'vy': obs[7]*20, 'can_double_jump': False, 'is_waiting_for_jump_key_release': False}
        self.ball = {'x': obs[8]*WALL_X, 'y': obs[9]*CEILING_Y, 'vx': obs[10]*20, 'vy': obs[11]*20}
        if reset_score:
            self.score_red = 0
            self.score_blue = 0
        self.time_steps = 0
        self.to_draw = []
        return self._get_internal_observation()

    def _update_player(self, player, keys):
        jump_failed = False
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
            else:
                jump_failed = True

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

        move_towards_ball = (not keys['left'] and keys['right'] and self.ball['x'] > player['x']) or (keys['left'] and not keys['right'] and self.ball['x'] < player['x'])

        return jump_failed, move_towards_ball

    def _update_ball(self):
        red_kicked = False
        blue_kicked = False
        ball_hit_ground = False
        ball_hit_ceiling = False
        ball_hit_left_wall = False
        ball_hit_right_wall = False

        def process_collision(player):
            dx = self.ball['x'] - player['x']
            self.ball['vx'] = (player['vx'] * abs(dx)) / 5 + dx / 5
            self.ball['vy'] = player['vy'] + 10

        red_ball_dist = np.hypot(self.ball['x'] - self.red['x'], self.ball['y'] - self.red['y'])
        blue_ball_dist = np.hypot(self.ball['x'] - self.blue['x'], self.ball['y'] - self.blue['y'])

        if red_ball_dist < 20:
            process_collision(self.red)
            red_kicked = True
        if blue_ball_dist < 20:
            process_collision(self.blue)
            blue_kicked = True

        self.ball['vy'] -= 1; self.ball['vx'] *= 0.97
        self.ball['x'] += self.ball['vx']; self.ball['y'] += self.ball['vy']

        if self.ball['x'] > WALL_X:
            self.ball['x'], self.ball['vx'] = np.sign(self.ball['x']) * WALL_X, self.ball['vx'] * -0.7
            ball_hit_right_wall = True

        if self.ball['x'] < -WALL_X:
            self.ball['x'], self.ball['vx'] = np.sign(self.ball['x']) * WALL_X, self.ball['vx'] * -0.7
            ball_hit_left_wall = True

        if self.ball['y'] < GROUND_Y:
            ball_hit_ground = True
            self.ball['y'], self.ball['vy'] = GROUND_Y, self.ball['vy'] * -0.7
        if self.ball['y'] > CEILING_Y:
            ball_hit_ceiling = True
            self.ball['y'], self.ball['vy'] = CEILING_Y, self.ball['vy'] * -0.7

        if abs(self.ball['x']) > 205 and self.ball['y'] > -40 and self.ball['y'] + self.ball['vy'] <= -40:
            self.ball['y'], self.ball['vy'] = -40, 5; self.ball['vx'] = -5 * np.sign(self.ball['x'])

        return (red_ball_dist, red_kicked), (blue_ball_dist, blue_kicked), (ball_hit_ground, ball_hit_ceiling, ball_hit_left_wall, ball_hit_right_wall)

    def step(self, red_keys, blue_keys):
        red_state, blue_state, game_state = {}, {}, {}
        red_state['jump_failed'], red_state['move_towards_ball'] = self._update_player(self.red, red_keys)
        blue_state['jump_failed'], blue_state['move_towards_ball'] = self._update_player(self.blue, blue_keys)
        (red_state['ball_dist'], red_state['kicked']), (blue_state['ball_dist'], blue_state['kicked']), (ball_hit_ground, ball_hit_ceiling, ball_hit_left_wall, ball_hit_right_wall) = self._update_ball()
        red_state['scored'], blue_state['scored'] = False, False
        red_state['x'], red_state['y'] = self.red['x']/WALL_X, self.red['y']/CEILING_Y
        game_state['ball_x'], game_state['ball_y'] = self.ball['x']/WALL_X, self.ball['y']/CEILING_Y

        if self.ball['y'] < -40:
            if self.ball['x'] > 210:
                self.score_red += 1
                red_state['scored'] = True
                self._reset_round()
            elif self.ball['x'] < -210:
                self.score_blue += 1
                blue_state['scored'] = True
                self._reset_round()

        terminated = (self.score_red >= 10 or self.score_blue >= 10)

        self.time_steps += 1
        truncated = self.time_steps >= 1800
        game_state['time_steps'] = self.time_steps
        game_state['ball_hit_ground'] = ball_hit_ground
        game_state['ball_hit_ceiling'] = ball_hit_ceiling
        game_state['ball_hit_left_wall'] = ball_hit_left_wall
        game_state['ball_hit_right_wall'] = ball_hit_right_wall

        return self._get_internal_observation(), (red_state, blue_state, game_state), terminated, truncated, {}

    def draw(self, shape, xc, yc):
        self.to_draw.append((shape, (xc*WALL_X, yc*CEILING_Y)))

    def render(self):
        if self.render_mode != 'human':
            return

        if self.screen is None:
            # First-time setup
            pygame.init()
            pygame.font.init()
            pygame.display.set_caption("Football RL")
            self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
            self.clock = pygame.time.Clock()

        self.screen.fill(COLOR_SKY)

        # Goals (drawn first)
        for side in [-1, 1]:
            color = COLOR_RED if side == -1 else COLOR_BLUE
            x_post = side * 230

            # --- Outline ---
            # Vertical post as a rectangle
            post_rect_outline_tl = self._s2p(x_post - 10, -50)
            pygame.draw.rect(self.screen, COLOR_BLACK, (post_rect_outline_tl[0], post_rect_outline_tl[1], int(20 * self.scale), int(self.scale * 120)))
            # Horizontal crossbar as a rectangle
            crossbar_x_start = x_post if side == 1 else x_post - 10
            crossbar_rect_outline_tl = self._s2p(crossbar_x_start, -40)
            pygame.draw.rect(self.screen, COLOR_BLACK, (crossbar_rect_outline_tl[0], crossbar_rect_outline_tl[1], int(10 * self.scale), int(20 * self.scale)))
            # Corner circle to smooth the join
            pygame.draw.circle(self.screen, COLOR_BLACK, self._s2p(x_post, -50), int(10 * self.scale))

            # --- Fill ---
            # Vertical fill
            post_rect_fill_tl = self._s2p(x_post - 8, -50)
            pygame.draw.rect(self.screen, color, (post_rect_fill_tl[0], post_rect_fill_tl[1], int(16 * self.scale), int(120 * self.scale)))
            # Horizontal fill
            crossbar_x_fill_start = x_post if side == 1 else x_post - 8
            crossbar_rect_fill_tl = self._s2p(crossbar_x_fill_start, -42)
            pygame.draw.rect(self.screen, color, (crossbar_rect_fill_tl[0], crossbar_rect_fill_tl[1], int(8 * self.scale), int(16 * self.scale)))
            # Corner circle fill
            pygame.draw.circle(self.screen, color, self._s2p(x_post, -50), int(8 * self.scale))

        # Ground (drawn over goal bottoms)
        pygame.draw.line(self.screen, COLOR_BLACK, self._s2p(-240, -170), self._s2p(240, -170), int(20 * self.scale))
        pygame.draw.line(self.screen, COLOR_GRASS, self._s2p(-240, -170), self._s2p(240, -170), int(16 * self.scale))

        # Accurate Boundary
        pygame.draw.rect(self.screen, COLOR_BLACK, (0, 0, self.width, self.height), int(2 * self.scale))

        # Players and ball
        pygame.draw.circle(self.screen, COLOR_BLACK, self._s2p(self.red['x'], self.red['y']), int(10 * self.scale)); pygame.draw.circle(self.screen, COLOR_RED, self._s2p(self.red['x'], self.red['y']), int(8 * self.scale))
        pygame.draw.circle(self.screen, COLOR_BLACK, self._s2p(self.blue['x'], self.blue['y']), int(10 * self.scale)); pygame.draw.circle(self.screen, COLOR_BLUE, self._s2p(self.blue['x'], self.blue['y']), int(8 * self.scale))
        pygame.draw.circle(self.screen, COLOR_BLACK, self._s2p(self.ball['x'], self.ball['y']), int(10 * self.scale)); pygame.draw.circle(self.screen, COLOR_WHITE, self._s2p(self.ball['x'], self.ball['y']), int(8 * self.scale))

        score_text = self.get_font().render(f"{self.score_red} - {self.score_blue}", True, COLOR_BLACK)
        self.screen.blit(score_text, (self.width/2 - score_text.get_width()/2, int(10 * self.scale)))

        for shape, (xc, yc) in self.to_draw:
            if shape == 'cross':
                pygame.draw.line(self.screen, COLOR_BLACK, self._s2p(xc - 7, yc - 7), self._s2p(xc + 7, yc + 7), 2)
                pygame.draw.line(self.screen, COLOR_BLACK, self._s2p(xc - 7, yc + 7), self._s2p(xc + 7, yc - 7), 2)
            else:
                raise NotImplementedError(f'Shape {shape} not implemented')

        pygame.display.flip()

    def close(self):
        """Shuts down the Pygame instance if it was created."""
        if self.screen is not None:
            pygame.quit()
            self.screen = None

if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((BASE_WIDTH, BASE_HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("Pen Football - Two Player")
    clock = pygame.time.Clock()
    game = FootballGame(screen)
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
        red_keys = { 'jump': keys[pygame.K_w], 'left': keys[pygame.K_a], 'right': keys[pygame.K_d] }
        blue_keys = { 'jump': keys[pygame.K_UP], 'left': keys[pygame.K_LEFT], 'right': keys[pygame.K_RIGHT] }
        _, _, terminated, _, _ = game.step(red_keys, blue_keys)
        if terminated:
            print(f"Game Over! Final Score: Red {game.score_red} - Blue {game.score_blue}")
            game.reset()
        game.render()
        clock.tick(TICK_RATE)
    pygame.quit()
