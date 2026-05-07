import pygame
from gymnasium import spaces
import gymnasium as gym
import numpy as np


# def u_func(x):
#     #− 1.018 × 10−6x4 + 7.563 × 10−5x3 − 0.001872x2 + 0.02022x + .3944.
#     return -1.018e-6 * x**4 + 7.563e-5 * x**3 - 0.001872 * x**2 + 0.02022 * x + 0.3944


# x = np.linspace(17, 30, 100)
# y = u_func(x)

# print(y.min(), y.max(), y.mean())
#Result: 0.484 0.533 0.508

# plt.plot(x, y)
# plt.xlabel('x')
# plt.ylabel('u(x)')
# plt.title('u_func')
# plt.grid(True)
# plt.show()

#Transition function for the Control room problem taken from: https://github.com/oxford-oxcav/fossil/blob/main/experiments/benchmarks/models.py CtrlRoomTemp class
#x(t) + τs (αe(Te − x(t))+ αH (Th − x(t))u(t)) 
tau = 5 * 60
alpha_e = 8 * 1e-3  # heat exchange room-external
temp_e = 15  # external temp
alpha_h = 3.6 * 1e-3  # heat exchange room-heater
temp_h = 55
def transition(x, u):
        #print(f"transition function called with x={x}, u={u}")
        return x + tau * (alpha_e * (temp_e - x) + alpha_h * (temp_h - x) * u)

TEMP_DISC = 400
CTRL_DISC = 100
MIN_TEMP = 15
MAX_TEMP = 35
TEMP_INTERVAL = (MAX_TEMP - MIN_TEMP) / TEMP_DISC
CTRL_INTERVAL = 1.0 / CTRL_DISC
class CtrlRoomTemp(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    INITIAL = (18.0, 19.0)
    VALID = (17.0, 28.0)
    def __init__(self, max_steps: int = 200, render_mode=None):
        super().__init__()
        self.observation_space = spaces.Box(
            low=np.array([MIN_TEMP], dtype=np.float32),
            high=np.array([MAX_TEMP], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.state = None
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([self.np_random.uniform(self.INITIAL[0], self.INITIAL[1])], dtype=np.float32)
        self.steps = 0
        return self.state.copy(), {}

    def step(self, action):
        #Sample from the intervals using the action probabilities and choose the mean of the interval as the action
        #u = np.random.choice(CTRL_DISC, p=action_prob) * CTRL_INTERVAL + 0.5 * CTRL_INTERVAL
                        
        x_next = np.clip(transition(self.state, action), MIN_TEMP, MAX_TEMP)

        self.state = x_next #np.array([x_next], dtype=np.float32)

        reward = 1.0 if self.VALID[0] <= x_next <= self.VALID[1] else -10.0

        self.steps += 1
        terminated = False
        truncated = self.steps >= self.max_steps

        return self.state.copy(), reward, terminated, truncated, {}

    def render(self):
        if not hasattr(self, "_pygame_initialized"):
            pygame.init()
            pygame.font.init()
            self._screen_width = 900
            self._screen_height = 220
            if self.render_mode == "human":
                self._screen = pygame.display.set_mode((self._screen_width, self._screen_height))
                pygame.display.set_caption("CtrlRoomTemp")
                self._clock = pygame.time.Clock()
            else:
                self._screen = pygame.Surface((self._screen_width, self._screen_height))
            self._font = pygame.font.SysFont(None, 24)
            self._pygame_initialized = True

        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    self._pygame_initialized = False
                    return

        screen = self._screen
        screen.fill((245, 245, 245))

        margin_x = 70
        scale_y = 120
        scale_h = 28
        usable_w = self._screen_width - 2 * margin_x

        def temp_to_x(temp):
            temp = float(np.clip(temp, MIN_TEMP, MAX_TEMP))
            ratio = (temp - MIN_TEMP) / (MAX_TEMP - MIN_TEMP)
            return int(margin_x + ratio * usable_w)

        # Colors
        axis_color = (40, 40, 40)
        valid_color = (170, 230, 170)
        initial_color = (120, 170, 255)
        state_color = (220, 60, 60)
        tick_color = (80, 80, 80)
        text_color = (20, 20, 20)

        # Base scale
        pygame.draw.rect(screen, (220, 220, 220), (margin_x, scale_y, usable_w, scale_h), border_radius=6)

        # Valid interval
        valid_x0 = temp_to_x(self.VALID[0])
        valid_x1 = temp_to_x(self.VALID[1])
        pygame.draw.rect(
            screen,
            valid_color,
            (valid_x0, scale_y, max(1, valid_x1 - valid_x0), scale_h),
            border_radius=6,
        )

        # Initial interval
        init_x0 = temp_to_x(self.INITIAL[0])
        init_x1 = temp_to_x(self.INITIAL[1])
        pygame.draw.rect(
            screen,
            initial_color,
            (init_x0, scale_y, max(1, init_x1 - init_x0), scale_h),
            border_radius=6,
        )

        # Border and axis line
        pygame.draw.rect(screen, axis_color, (margin_x, scale_y, usable_w, scale_h), width=2, border_radius=6)
        pygame.draw.line(screen, axis_color, (margin_x, scale_y + scale_h // 2), (margin_x + usable_w, scale_y + scale_h // 2), 2)

        # Min/max ticks and labels
        for t in [MIN_TEMP, MAX_TEMP]:
            x = temp_to_x(t)
            pygame.draw.line(screen, tick_color, (x, scale_y - 8), (x, scale_y + scale_h + 8), 2)
            label = self._font.render(f"{t:.0f}", True, text_color)
            screen.blit(label, (x - label.get_width() // 2, scale_y + scale_h + 14))

        # Interval labels
        valid_label = self._font.render(f"valid [{self.VALID[0]:.1f}, {self.VALID[1]:.1f}]", True, text_color)
        init_label = self._font.render(f"initial [{self.INITIAL[0]:.1f}, {self.INITIAL[1]:.1f}]", True, text_color)
        screen.blit(valid_label, (margin_x, 20))
        screen.blit(init_label, (margin_x, 48))

        pygame.draw.rect(screen, valid_color, (margin_x + 210, 24, 18, 12))
        pygame.draw.rect(screen, initial_color, (margin_x + 210, 52, 18, 12))

        # Current state bar
        if self.state is not None:
            state_temp = float(self.state[0])
            state_x = temp_to_x(state_temp)
            bar_w = 8
            bar_top = scale_y - 22
            bar_h = scale_h + 44
            pygame.draw.rect(
            screen,
            state_color,
            (state_x - bar_w // 2, bar_top, bar_w, bar_h),
            border_radius=3,
            )
            state_label = self._font.render(f"T = {state_temp:.2f}", True, state_color)
            screen.blit(state_label, (state_x - state_label.get_width() // 2, bar_top - 28))

        title = self._font.render("Room temperature scale", True, text_color)
        screen.blit(title, (margin_x, 85 - title.get_height()))

        if self.render_mode == "rgb_array":
            return pygame.surfarray.array3d(self._screen).transpose(1, 0, 2)

        pygame.display.flip()
        self._clock.tick(30)
        return None

    def close(self):
        if hasattr(self, "_pygame_initialized") and self._pygame_initialized:
            pygame.quit()
            self._pygame_initialized = False
    