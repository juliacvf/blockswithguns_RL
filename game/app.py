"""Pygame application: menu, topdown mode, split-screen 3D polygon mode."""

from __future__ import annotations

import math
import os
import sys

import pygame

from bots import BOT_NAMES, make_bot, make_view
from core import constants as C
from core.engine import Action, Engine

from .animations import Flash, ParticleSystem, ScreenShake
from .hud import Hud
from .render_poly import SplitScreenPoly
from .render_topdown import Minimap, TopdownRenderer
from . import textures as T

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "sfx")


class Sounds:
    """Lazy sound bank; silently disables itself when audio is unavailable."""

    def __init__(self, enabled: bool = True):
        self.enabled = False
        self.bank: dict[str, pygame.mixer.Sound] = {}
        if not enabled:
            return
        try:
            pygame.mixer.init(22050, -16, 1, 512)
            for f in os.listdir(ASSETS):
                if f.endswith(".wav"):
                    self.bank[f[:-4]] = pygame.mixer.Sound(os.path.join(ASSETS, f))
            self.enabled = True
        except (pygame.error, FileNotFoundError):
            self.enabled = False

    def play(self, name: str, vol: float = 1.0) -> None:
        if self.enabled and name in self.bank:
            self.bank[name].set_volume(vol)
            self.bank[name].play()

    def music(self, name: str, vol: float = 0.6) -> None:
        if self.enabled and name in self.bank:
            pygame.mixer.music.stop()
            snd = os.path.join(ASSETS, name + ".wav")
            try:
                pygame.mixer.music.load(snd)
                pygame.mixer.music.set_volume(vol)
                pygame.mixer.music.play(-1)
            except pygame.error:
                pass

    def stop_music(self) -> None:
        if self.enabled:
            pygame.mixer.music.stop()


def _mouse_inside_window() -> bool:
    """Whether the cursor is over the game window (mouse.md focus check).

    Headless (dummy) video drivers report no mouse focus; treat them as
    always inside so headless tooling keeps working.
    """
    if not pygame.display.get_init():
        return True
    return pygame.display.get_driver() == "dummy" or pygame.mouse.get_focused()


class HumanController:
    """Keyboard/mouse input -> Action for player 1 (WASD + mouse).

    Shooting: left mouse click or the E key (hold E for full-auto at the
    engine's cooldown rate). In topdown mode the aim is computed with
    atan2 toward the cursor position, tracked only while the cursor is
    inside the window. In poly mode the mouse turns the view horizontally:
    the app measures the cursor's x offset from an anchor at the screen
    center (projecting the mouse position onto the x axis, so vertical
    motion is ignored) and accumulates it in `pending_rel`, re-centering
    the cursor every frame so turning never stalls at the screen edge;
    the LEFT/RIGHT arrow keys turn the view as well.
    """

    TURN_SPEED = 3.0  # rad/s for arrow-key turning in poly mode

    def __init__(self, mode: str):
        self.mode = mode
        self._shoot = False
        self.pending_rel = 0.0

    def on_event(self, ev: pygame.event.Event) -> None:
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            self._shoot = True

    def build(self, engine: Engine, ppc: float = 4.0, origin=(0, 0)) -> Action:
        pl = engine.players[0]
        keys = pygame.key.get_pressed()
        shoot = self._shoot or bool(keys[pygame.K_e])
        self._shoot = False
        if self.mode == "topdown":
            mx = (keys[pygame.K_d] - keys[pygame.K_a])
            my = (keys[pygame.K_s] - keys[pygame.K_w])
            # aim follows the cursor only while it is inside the window;
            # outside, get_pos() freezes at the edge (see mouse.md)
            aim = pl.aim
            if _mouse_inside_window():
                wx, wy = pygame.mouse.get_pos()
                wx = (wx - origin[0]) / ppc
                wy = (wy - origin[1]) / ppc
                aim = math.atan2(wy - pl.y, wx - pl.x)
        else:
            # consume the mouse delta accumulated since the last tick
            rel = self.pending_rel
            self.pending_rel = 0.0
            aim = pl.aim + rel * 0.004
            # LEFT/RIGHT arrows turn the view (RIGHT = clockwise, like mouse right)
            turn = keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]
            aim += turn * self.TURN_SPEED * C.FIXED_DT
            # W/S move along aim, A/D strafe
            fwd = keys[pygame.K_w] - keys[pygame.K_s]
            side = keys[pygame.K_d] - keys[pygame.K_a]
            mx = math.cos(pl.aim) * fwd - math.sin(pl.aim) * side
            my = math.sin(pl.aim) * fwd + math.cos(pl.aim) * side
        return Action(move_x=mx, move_y=my, aim=aim, shoot=shoot)


class BotController:
    def __init__(self, name: str, player: int, seed: int = 0):
        self.bot = make_bot(name, seed=seed)
        self.player = player

    def reset(self) -> None:
        self.bot.reset()

    def build(self, engine: Engine, ppc: float = 4.0, origin=(0, 0)) -> Action:
        return self.bot.act(make_view(engine, self.player))


MODES = ["poly", "topdown"]
PLAYER_SETUPS = ["HUMAN VS ALGO", "ALGO VS ALGO"]


class App:
    def __init__(self, sound: bool = True, fps: int = 60):
        pygame.init()
        pygame.display.set_caption("Blocks With Guns RL")
        self.screen = pygame.display.set_mode((1024, 812))
        self.clock = pygame.time.Clock()
        self.fps = fps
        self.sounds = Sounds(enabled=sound)
        self.hud = Hud()
        self.font = pygame.font.Font(None, 28)
        self.big = pygame.font.Font(None, 72)
        self.state = "menu"
        self.menu_idx = 0
        self.cfg = {"mode": "poly", "players": "HUMAN VS ALGO",
                    "algo1": "astar", "algo2": "markov",
                    "seed": 1, "scale": 24.0, "threshold": 0.62, "octaves": 3}
        self.engine: Engine | None = None
        self.controllers: list = []
        self.topdown: TopdownRenderer | None = None
        self.poly: SplitScreenPoly | None = None
        self.minimap: Minimap | None = None
        self.particles = ParticleSystem()
        self.shakes = [ScreenShake(), ScreenShake()]
        self.flashes = [Flash(), Flash()]
        self.recoil = [0.0, 0.0]
        self.bob = [0.0, 0.0]
        self._acc = 0.0
        self._mouse_anchor_x = 0.0
        self.menu_bg = self._build_menu_bg()
        self.cobble_bg = self._build_cobble_bg()
        self._build_title()
        self._build_formulas()

    def _build_cobble_bg(self) -> pygame.Surface:
        """Tiled cobblestone backdrop for the play screen."""
        w, h = self.screen.get_size()
        tile = pygame.transform.scale(T.cobble_texture(), (48, 48))
        bg = pygame.Surface((w, h))
        for x in range(0, w, 48):
            for y in range(0, h, 48):
                bg.blit(tile, (x, y))
        dark = pygame.Surface((w, h), pygame.SRCALPHA)
        dark.fill((10, 12, 10, 70))  # slight darken so the views pop
        bg.blit(dark, (0, 0))
        return bg

    # ------------------------------------------------------------------ #
    TITLE_TEXT = "BLOCKS WITH GUNS"

    def _build_title(self) -> None:
        """Shiny metallic gold title: vertical gradient clipped to the glyphs."""
        font = pygame.font.Font(None, 104)
        mask = font.render(self.TITLE_TEXT, True, (255, 255, 255))
        w, h = mask.get_size()

        def lerp(a, b, f):
            return tuple(int(a[i] + (b[i] - a[i]) * f) for i in range(3))

        grad = pygame.Surface((w, h), pygame.SRCALPHA)
        top, mid, bot = (255, 248, 200), (255, 210, 84), (188, 110, 26)
        for y in range(h):
            f = y / max(1, h - 1)
            col = lerp(top, mid, f * 2) if f < 0.5 else lerp(mid, bot, (f - 0.5) * 2)
            pygame.draw.line(grad, col, (0, y), (w, y))
        grad.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)

        base = pygame.Surface((w + 8, h + 8), pygame.SRCALPHA)
        outline = font.render(self.TITLE_TEXT, True, (46, 30, 8))
        for dx in range(0, 9, 4):
            for dy in range(0, 9, 4):
                base.blit(outline, (dx, dy))
        base.blit(grad, (4, 4))
        self._title = base
        self._title_mask = mask
        # text-shaped soft shadow (not a rectangle)
        shadow = font.render(self.TITLE_TEXT, True, (8, 10, 8))
        shadow.set_alpha(110)
        self._title_shadow = shadow

    def _draw_title(self, t: float) -> None:
        w = self.screen.get_width()
        x, y = (w - self._title.get_width()) // 2, 84
        self.screen.blit(self._title_shadow, (x + 9, y + 11))
        self.screen.blit(self._title, (x, y))
        # animated diagonal shine, soft single band clipped to glyphs
        mw, mh = self._title_mask.get_size()
        band = 55
        span = mw + 2 * band + 240
        pos = (t * 150.0) % span - band - 120
        shine = pygame.Surface((mw, mh), pygame.SRCALPHA)
        pygame.draw.polygon(shine, (255, 255, 255, 26),
                            [(pos - band, mh + 20), (pos + band, mh + 20),
                             (pos + band + 45, -20), (pos - band + 45, -20)])
        pygame.draw.polygon(shine, (255, 255, 255, 34),
                            [(pos - band // 2, mh + 20), (pos + band // 2, mh + 20),
                             (pos + band // 2 + 45, -20), (pos - band // 2 + 45, -20)])
        shine.blit(self._title_mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        # plain alpha blit: ADD would ignore per-pixel alpha and slab over
        # the background (font AA leaves RGB=255 in zero-alpha pixels)
        self.screen.blit(shine, (x + 4, y + 4))

    # RL formulas graffiti on the menu background (glyphs limited to the
    # ones pygame's stock font actually has: no arrows / nabla).
    FORMULAS = [
        "Q*(s,a) = r + γ·max_a' Q*(s',a')",
        "V*(s) = max_a Σ p[r + γV*(s')]",
        "G_t = Σ_k γ^k · r_(t+k+1)",
        "δ = r + γV(s') - V(s)",
        "π(a|s) = softmax(Q(s,a)/τ)",
        "θ += α · δ · log π_θ(a|s)",
        "A(s,a) = Q(s,a) - V(s)",
        "V_π(s) = E_π[ G_t | s ]",
    ]
    # (formula index, x center fraction, y fraction, drift phase)
    # kept in the side strips / bottom band, clear of title and panel
    FORMULA_SPOTS = [
        (0, 0.135, 0.310, 0.0), (1, 0.865, 0.310, 1.3),
        (2, 0.130, 0.430, 2.1), (3, 0.870, 0.430, 0.6),
        (4, 0.130, 0.550, 3.4), (5, 0.870, 0.550, 2.7),
        (6, 0.290, 0.845, 1.8), (7, 0.720, 0.845, 4.2),
    ]

    def _build_formulas(self) -> None:
        font = pygame.font.Font(None, 22)
        self._formula_imgs = []
        for s in self.FORMULAS:
            img = font.render(s, True, (214, 232, 196))
            img.set_alpha(96)
            self._formula_imgs.append(img)

    def _draw_formulas(self, t: float) -> None:
        w, h = self.screen.get_size()
        for idx, fx, fy, phase in self.FORMULA_SPOTS:
            img = self._formula_imgs[idx]
            yoff = math.sin(t * 0.7 + phase) * 4.0
            self.screen.blit(img, (fx * w - img.get_width() / 2, fy * h + yoff))

    # ------------------------------------------------------------------ #
    def _build_menu_bg(self) -> pygame.Surface:
        w, h = self.screen.get_size()
        eng = Engine(seed=3, scale=30, threshold=0.6)
        r = TopdownRenderer(eng, px_per_cell=4)
        map_surf = pygame.Surface((r.size, r.size))
        r.draw(map_surf, tick=0)
        # uniform cover-scale (blocks stay square), then center-crop
        factor = max(w / r.size, h / r.size)
        scaled = pygame.transform.scale(
            map_surf, (int(r.size * factor) + 1, int(r.size * factor) + 1))
        s = pygame.Surface((w, h))
        s.blit(scaled, ((w - scaled.get_width()) // 2, (h - scaled.get_height()) // 2))
        dark = pygame.Surface(s.get_size(), pygame.SRCALPHA)
        dark.fill((20, 26, 20, 120))
        s.blit(dark, (0, 0))
        return s

    # ------------------------------------------------------------------ #
    def start_match(self) -> None:
        cfg = self.cfg
        self.engine = Engine(seed=cfg["seed"], scale=cfg["scale"],
                             octaves=cfg["octaves"], threshold=cfg["threshold"])
        self.human_p0 = cfg["players"] == "HUMAN VS ALGO"
        if self.human_p0:
            self.controllers = [HumanController(cfg["mode"])]
        else:
            self.controllers = [BotController(cfg["algo1"], 0, seed=cfg["seed"] * 7 + 3)]
        self.controllers.append(BotController(cfg["algo2"], 1, seed=cfg["seed"] * 7 + 1))
        self.topdown = TopdownRenderer(self.engine, px_per_cell=6)
        # split-screen views: full width, P1 top half, P2 bottom half
        sw, sh = self.screen.get_size()
        self.view_margin = 12
        self.view_w = sw - 2 * self.view_margin
        self.view_h = (sh - 3 * self.view_margin) // 2
        self.poly = SplitScreenPoly(self.engine, view_w=self.view_w,
                                    view_h=self.view_h)
        self.minimap = Minimap(self.engine, size=190)
        self.particles = ParticleSystem()
        self.retreat_until = [0.0, 0.0]   # per-player RETREAT! text end time
        self._acc = 0.0
        self.state = "play"
        self.sounds.stop_music()
        self.sounds.music("battle_theme", 0.5)
        if cfg["mode"] == "poly" and self.human_p0:
            pygame.event.set_grab(True)
            pygame.mouse.set_visible(False)
            pygame.mouse.set_pos(sw // 2, sh // 2)
            self._mouse_anchor_x = pygame.mouse.get_pos()[0]
        elif self.human_p0:
            # topdown: hide the OS cursor, we draw our own crosshair
            pygame.mouse.set_visible(False)

    def pname(self, idx: int) -> str:
        """Display name for player idx ('PLAYER 1' when P1 is human)."""
        if idx == 0 and self.human_p0:
            return "PLAYER 1"
        return self.cfg["algo1" if idx == 0 else "algo2"].upper()

    def end_to_menu(self) -> None:
        self.state = "menu"
        pygame.event.set_grab(False)
        pygame.mouse.set_visible(True)
        self.sounds.stop_music()
        self.sounds.music("menu_theme", 0.4)

    # ------------------------------------------------------------------ #
    MENU_ITEMS = ["mode", "players", "algo1", "algo2", "seed", "scale",
                  "threshold", "START"]

    def _menu_handle(self, ev: pygame.event.Event) -> None:
        if ev.type != pygame.KEYDOWN:
            return
        if ev.key in (pygame.K_UP, pygame.K_w):
            self.menu_idx = (self.menu_idx - 1) % len(self.MENU_ITEMS)
            self.sounds.play("ui_hover", 0.4)
        elif ev.key in (pygame.K_DOWN, pygame.K_s):
            self.menu_idx = (self.menu_idx + 1) % len(self.MENU_ITEMS)
            self.sounds.play("ui_hover", 0.4)
        elif ev.key in (pygame.K_LEFT, pygame.K_RIGHT, pygame.K_a, pygame.K_d):
            d = 1 if ev.key in (pygame.K_RIGHT, pygame.K_d) else -1
            item = self.MENU_ITEMS[self.menu_idx]
            if item == "mode":
                i = (MODES.index(self.cfg["mode"]) + d) % len(MODES)
                self.cfg["mode"] = MODES[i]
            elif item == "players":
                i = (PLAYER_SETUPS.index(self.cfg["players"]) + d) % len(PLAYER_SETUPS)
                self.cfg["players"] = PLAYER_SETUPS[i]
            elif item == "algo1" and self.cfg["players"] == "ALGO VS ALGO":
                i = (BOT_NAMES.index(self.cfg[item]) + d) % len(BOT_NAMES)
                self.cfg[item] = BOT_NAMES[i]
            elif item == "algo2":
                i = (BOT_NAMES.index(self.cfg[item]) + d) % len(BOT_NAMES)
                self.cfg[item] = BOT_NAMES[i]
            elif item == "seed":
                self.cfg["seed"] = max(0, self.cfg["seed"] + d)
            elif item == "scale":
                self.cfg["scale"] = min(60.0, max(8.0, self.cfg["scale"] + d * 2.0))
            elif item == "threshold":
                self.cfg["threshold"] = min(0.8, max(0.45, round(self.cfg["threshold"] + d * 0.02, 2)))
            self.sounds.play("ui_hover", 0.4)
        elif ev.key == pygame.K_r:
            self.cfg["seed"] = int(pygame.time.get_ticks()) % 100000
        elif ev.key in (pygame.K_RETURN, pygame.K_SPACE):
            if self.MENU_ITEMS[self.menu_idx] == "START":
                self.start_match()

    def _menu_draw(self) -> None:
        t = pygame.time.get_ticks() / 1000.0
        self.screen.blit(self.menu_bg, (0, 0))
        self._draw_formulas(t)
        self._draw_title(t)
        self.hud.text(self.screen, "R L   E D I T I O N", 512, 196, (170, 230, 255), center=True)
        # framed panel behind the options (rounded corners)
        panel = pygame.Surface((500, 420), pygame.SRCALPHA)
        pygame.draw.rect(panel, (14, 18, 13, 175), panel.get_rect(), border_radius=18)
        pygame.draw.rect(panel, (255, 217, 90, 120), panel.get_rect(), 3, border_radius=18)
        self.screen.blit(panel, (262, 232))
        y = 270
        labels = {
            "mode": f"MODE        < {self.cfg['mode'].upper()} >",
            "players": f"PLAYERS     < {self.cfg['players']} >",
            "algo1": ("PLAYER 1    < HUMAN >   [LOCKED]"
                      if self.cfg["players"] == "HUMAN VS ALGO"
                      else f"ALGO 1      < {self.cfg['algo1'].upper()} >"),
            "algo2": f"ALGO 2      < {self.cfg['algo2'].upper()} >",
            "seed": f"MAP SEED    < {self.cfg['seed']} >   (R = random)",
            "scale": f"NOISE SCALE < {self.cfg['scale']:.0f} >",
            "threshold": f"WALL DENS.  < {self.cfg['threshold']:.2f} >",
            "START": ">>> START MATCH <<<",
        }
        for i, item in enumerate(self.MENU_ITEMS):
            sel = i == self.menu_idx
            locked = item == "algo1" and self.cfg["players"] == "HUMAN VS ALGO"
            rect = pygame.Rect(292, y - 14, 440, 34)
            overlay = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
            if locked:
                color = (145, 150, 145)
                pygame.draw.rect(overlay, (70, 75, 70, 115), overlay.get_rect(), border_radius=16)
                pygame.draw.rect(overlay, (135, 140, 135, 100), overlay.get_rect(), 1, border_radius=16)
            elif sel:
                color = (50, 40, 12)
                pygame.draw.rect(overlay, (255, 215, 90, 220), overlay.get_rect(), border_radius=16)
                pygame.draw.rect(overlay, (255, 240, 150), overlay.get_rect(), 2, border_radius=16)
            else:
                color = (220, 220, 218)
                pygame.draw.rect(overlay, (255, 255, 255, 22), overlay.get_rect(), border_radius=16)
                pygame.draw.rect(overlay, (255, 255, 255, 40), overlay.get_rect(), 1, border_radius=16)
            self.screen.blit(overlay, rect.topleft)
            self.hud.text(self.screen, labels[item], 512, y, color, center=True)
            y += 48
        hint = "WASD move  |  mouse aim  |  click / E shoot  |  arrows = options / turning  |  Q quit"
        self.hud.text(self.screen, hint, 512, 736, (180, 200, 180), center=True)

    # ------------------------------------------------------------------ #
    def _play_events_to_feedback(self, events: list[tuple]) -> None:
        for e in events:
            kind = e[0]
            if kind == "shot":
                self.sounds.play("shot", 0.25)
                self.recoil[e[1]] = 1.0
            elif kind == "hit":
                victim, shooter = e[1], e[2]
                self.sounds.play("hit", 0.5)
                self.flashes[victim].trigger()
                self.shakes[victim].add(0.6)
                pl = self.engine.players[victim]
                self.particles.burst(pl.x, pl.y, (255, 90, 90), count=16)
            elif kind == "heat_hit":
                victim = e[1]
                self.sounds.play("hit", 0.4)
                self.flashes[victim].trigger(color=(255, 120, 40), alpha=80)
                self.shakes[victim].add(0.4)
                pl = self.engine.players[victim]
                self.particles.burst(pl.x, pl.y, (255, 140, 60), count=12)
            elif kind == "retreat":
                victim, rx, ry = e[1], e[2], e[3]
                self.sounds.play("swap", 0.6)
                self.retreat_until[victim] = self.engine.time + 2.0
                self.flashes[victim].trigger(color=(255, 255, 255), alpha=70)
                self.particles.burst(rx, ry, (200, 200, 255), count=22, speed=6)
            elif kind == "pickup":
                pk, idx = e[1], e[2]
                if pk == "LIFE":
                    self.sounds.play("heart_pickup", 0.5)
                elif pk == "QUICKSHOT":
                    self.sounds.play("quickshot", 0.5)
                elif pk == "SWAP":
                    self.sounds.play("swap", 0.5)
                else:
                    self.sounds.play("powerup", 0.5)
                pl = self.engine.players[idx]
                self.particles.burst(pl.x, pl.y, T.POWERUP_COLORS[pk], count=18)
                self.flashes[idx].trigger(color=(255, 255, 255), alpha=50)
            elif kind == "powerup_spawn":
                self.particles.burst(e[2], e[3], (255, 255, 200), count=10, speed=3)
            elif kind == "gameover":
                self._on_gameover(e[1])

    def _on_gameover(self, winner: int) -> None:
        self.state = "gameover"
        pygame.event.set_grab(False)
        pygame.mouse.set_visible(True)
        self.sounds.stop_music()
        if winner == -1:
            self.sounds.play("defeat", 0.5)
        elif not self.human_p0 or winner == 0:
            self.sounds.play("victory", 0.5)
        else:
            self.sounds.play("defeat", 0.5)
        pl = self.engine.players[1 - winner] if winner in (0, 1) else self.engine.players[0]
        self.particles.burst(pl.x, pl.y, (255, 120, 120), count=40, speed=9, life=0.9)

    # ------------------------------------------------------------------ #
    def run(self) -> None:
        self.sounds.music("menu_theme", 0.4)
        while True:
            dt = self.clock.tick(self.fps) / 1000.0
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit(0)
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_q:
                    pygame.quit()
                    sys.exit(0)
                if self.state == "menu":
                    self._menu_handle(ev)
                elif self.state == "play":
                    if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                        self.end_to_menu()
                    for c in self.controllers:
                        if isinstance(c, HumanController):
                            c.on_event(ev)
                elif self.state == "gameover":
                    if ev.type == pygame.KEYDOWN and ev.key in (pygame.K_RETURN, pygame.K_ESCAPE):
                        self.end_to_menu()

            if self.state == "menu":
                self._menu_draw()
            elif self.state == "play":
                self._play_step(dt)
            elif self.state == "gameover":
                self._gameover_draw()
            pygame.display.flip()

    # ------------------------------------------------------------------ #
    def _play_step(self, dt: float) -> None:
        eng = self.engine
        # first-person mouse: project the cursor position onto the x axis —
        # only the horizontal offset from the anchor turns the camera, and
        # vertical motion is discarded so it can't interfere. Re-center the
        # cursor every frame so turning never stalls at the screen edge; the
        # anchor reads back the real position in case the warp was ignored.
        if self.cfg["mode"] == "poly" and self.human_p0:
            sw, sh = self.screen.get_size()
            ctrl = self.controllers[0]
            ctrl.pending_rel += pygame.mouse.get_pos()[0] - self._mouse_anchor_x
            pygame.mouse.set_pos(sw // 2, sh // 2)
            self._mouse_anchor_x = pygame.mouse.get_pos()[0]
        self._acc = min(self._acc + dt, 5 * C.FIXED_DT)
        while self._acc >= C.FIXED_DT and eng.winner is None:
            self._acc -= C.FIXED_DT
            ppc = 6.0
            # topdown mouse aim needs the map's on-screen offset
            size = C.GRID_SIZE * ppc
            origin = ((self.screen.get_width() - size) // 2,
                      (self.screen.get_height() - size) // 2)
            acts = tuple(c.build(eng, ppc=ppc, origin=origin) for c in self.controllers)
            events = eng.step(acts)
            self._play_events_to_feedback(events)
            # walk bob phase for moving players
            for i, a in enumerate(acts):
                if abs(a.move_x) > 0.05 or abs(a.move_y) > 0.05:
                    self.bob[i] += 0.35
        self.particles.update(dt)
        for i in range(2):
            self.shakes[i].update(dt)
            self.flashes[i].update(dt)
            self.recoil[i] = max(0.0, self.recoil[i] - dt * 6)
        self._play_draw()

    def _play_draw(self) -> None:
        eng = self.engine
        self.screen.blit(self.cobble_bg, (0, 0))
        sw, sh = self.screen.get_size()
        if self.cfg["mode"] == "topdown":
            ppc = 6
            size = C.GRID_SIZE * ppc
            ox = (sw - size) // 2
            oy = (sh - size) // 2
            shx, shy = self.shakes[0].offset()
            self.topdown.draw(self.screen, self.particles, tick=eng.tick, offset=(ox + shx, oy + shy))
            self.hud.draw_hearts(self.screen, 30, 30, eng.players[0].lives,
                                 pulse=eng.time - eng.players[0].last_hit_time < 0.4, tick=eng.tick)
            self.hud.draw_hearts(self.screen, 994, 30, eng.players[1].lives,
                                 align_right=True,
                                 pulse=eng.time - eng.players[1].last_hit_time < 0.4, tick=eng.tick)
            self.hud.draw_powerups(self.screen, 30, 62, eng.players[0])
            self.hud.draw_powerups(self.screen, 830, 62, eng.players[1])
            self.hud.text(self.screen, self.pname(0), 30, 8, (150, 200, 255))
            self.hud.text(self.screen, self.pname(1), 880, 8, (255, 170, 160))
            self.flashes[0].draw(self.screen)
            # "RETREAT!" on the side of the player who used the retreat
            for i, (nx, ny) in enumerate(((30, 90), (880, 90))):
                if eng.time < self.retreat_until[i]:
                    self.hud.text(self.screen, "RETREAT!", nx, ny, (255, 255, 255))
        else:
            vw, vh = self.view_w, self.view_h
            m = self.view_margin
            origins = ((m, m), (m, 2 * m + vh))
            self.poly.render(self.screen, origins, eng.tick,
                                recoil=(self.recoil[0], self.recoil[1]),
                                bob=(self.bob[0], self.bob[1]))
            for i, (ox, oy) in enumerate(origins):
                # hearts + powerups per view
                self.hud.draw_hearts(self.screen, ox + 10, oy + 10, eng.players[i].lives,
                                     pulse=eng.time - eng.players[i].last_hit_time < 0.4,
                                     tick=eng.tick)
                self.hud.draw_powerups(self.screen, ox + 10, oy + 40, eng.players[i])
                pl = eng.players[i]
                frac = min(1.0, pl.cooldown / pl.shoot_cooldown) if pl.shoot_cooldown else 0
                self.hud.draw_cooldown(self.screen, (ox + vw // 2, oy + vh // 2), frac)
                view_rect = pygame.Rect(ox, oy, vw, vh)
                sub = self.screen.subsurface(view_rect)
                self.flashes[i].draw(sub)
                # "RETREAT!" inside the view of the player who used it
                if eng.time < self.retreat_until[i]:
                    self.hud.text(self.screen, "RETREAT!", ox + 10, oy + 70, (255, 255, 255))
                # heat zone warning
                if eng.in_heat(pl.x, pl.y):
                    pygame.draw.rect(self.screen, (255, 70, 40), view_rect, 4)
                    left = max(0.0, C.HEAT_DAMAGE_PERIOD - pl.heat_timer)
                    self.hud.text(self.screen,
                                  f"HEAT ZONE! -1 LIFE IN {left:.1f}s",
                                  ox + vw // 2, oy + 28, (255, 130, 90), center=True)
            mm = self.minimap.size
            self.minimap.draw(self.screen, (sw - mm - m, sh - mm - m))
            self.hud.text(self.screen, self.pname(0), origins[0][0] + 10, origins[0][1] + vh - 28, (150, 200, 255))
            self.hud.text(self.screen, self.pname(1), origins[1][0] + 10, origins[1][1] + vh - 28, (255, 170, 160))
            self.hud.text(self.screen, "minimap", sw - mm - m + 4, sh - mm - m - 16, (200, 200, 210))
        self._draw_mouse_tracking()

    def _draw_mouse_tracking(self) -> None:
        """Custom crosshair + 'mouse left the window' alert (see mouse.md).

        The OS cursor stays hidden while it is over the window; when it
        slips onto another monitor the game shows it again plus a warning
        overlay, and the topdown aim freezes instead of sticking at the
        window edge. Clicks re-focus the window.
        """
        if self.state != "play" or not self.human_p0:
            return
        focused = _mouse_inside_window()
        pygame.mouse.set_visible(not focused)
        if focused:
            if self.cfg["mode"] == "topdown":
                mx, my = pygame.mouse.get_pos()
                col = (255, 255, 255)
                pygame.draw.circle(self.screen, col, (mx, my), 10, 2)
                pygame.draw.line(self.screen, col, (mx - 15, my), (mx + 15, my))
                pygame.draw.line(self.screen, col, (mx, my - 15), (mx, my + 15))
        else:
            w, h = self.screen.get_size()
            overlay = pygame.Surface((w, h), pygame.SRCALPHA)
            overlay.fill((200, 0, 0, 90))
            self.screen.blit(overlay, (0, 0))
            self.hud.text(self.screen, "MOUSE LEFT THE GAME - CLICK TO RETURN",
                          w // 2, h // 2, (255, 255, 255), center=True)

    # ------------------------------------------------------------------ #
    def _gameover_draw(self) -> None:
        self._play_draw()
        overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        overlay.fill((10, 10, 14, 170))
        self.screen.blit(overlay, (0, 0))
        w = self.engine.winner
        if w == -1:
            msg, color = "DRAW!", (240, 240, 200)
        elif w == 0:
            msg, color = f"{self.pname(0)} WINS!", (150, 220, 255)
        else:
            msg, color = f"{self.pname(1)} WINS!", (255, 160, 150)
        self.hud.text(self.screen, msg, 512, 380, color, big=True, center=True)
        self.hud.text(self.screen, "press ENTER for menu", 512, 440, (230, 230, 230), center=True)
