from __future__ import annotations

from multiprocessing import Queue
from queue import Empty
import random

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
from scipy.signal import butter, lfilter, lfilter_zi


CHANNEL_NAMES = ["AF8", "AF7", "CHEEK_R", "CHEEK_L", "EAR_R", "AFz", "BROW_L", "NOSE"]
GAME_WIDTH = 520
GAME_HEIGHT = 420


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    return butter(order, [low, high], btype="band")


class GameCanvas(QtWidgets.QWidget):
    title = "Game"
    control_hint = "Keyboard fallback: Space/arrow keys. Press R to reset."
    accent = "#7cff6b"

    def __init__(self):
        super().__init__()
        self.setMinimumSize(GAME_WIDTH, GAME_HEIGHT)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.score = 0
        self.status = "Ready"

    def handle_action(self, action: str):
        pass

    def handle_key(self, key: int):
        if key == QtCore.Qt.Key_R:
            self.reset()

    def step(self):
        pass

    def reset(self):
        self.score = 0
        self.status = "Ready"

    def draw_header(self, painter: QtGui.QPainter):
        panel = QtCore.QRectF(14, 12, GAME_WIDTH - 28, 62)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(2, 8, 6, 230))
        painter.drawRoundedRect(panel, 16, 16)
        painter.setPen(QtGui.QPen(QtGui.QColor(self.accent), 1.2))
        painter.drawRoundedRect(panel.adjusted(0.5, 0.5, -0.5, -0.5), 16, 16)

        painter.setPen(QtGui.QColor("#f8fafc"))
        painter.setFont(QtGui.QFont("Arial", 15, QtGui.QFont.Bold))
        painter.drawText(28, 34, self.title)

        score_rect = QtCore.QRectF(GAME_WIDTH - 116, 21, 86, 28)
        painter.setBrush(QtGui.QColor(124, 255, 107, 32))
        painter.setPen(QtGui.QPen(QtGui.QColor(self.accent), 1.5))
        painter.drawRoundedRect(score_rect, 14, 14)
        painter.setPen(QtGui.QColor("#dcfce7"))
        painter.setFont(QtGui.QFont("Arial", 11, QtGui.QFont.Bold))
        painter.drawText(score_rect, QtCore.Qt.AlignCenter, f"score {self.score}")

        painter.setPen(QtGui.QColor("#b9c8bd"))
        painter.setFont(QtGui.QFont("Arial", 9))
        subtitle = self.status
        subtitle_rect = QtCore.QRectF(28, 43, GAME_WIDTH - 164, 18)
        painter.drawText(subtitle_rect, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, self.elide(painter, subtitle, subtitle_rect.width()))

    def elide(self, painter: QtGui.QPainter, text: str, width: float):
        return painter.fontMetrics().elidedText(text, QtCore.Qt.ElideRight, max(12, int(width)))

    def draw_caption(self, painter: QtGui.QPainter, text: str):
        caption = QtCore.QRectF(18, GAME_HEIGHT - 42, GAME_WIDTH - 36, 26)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(2, 8, 6, 225))
        painter.drawRoundedRect(caption, 13, 13)
        painter.setPen(QtGui.QColor("#dcfce7"))
        painter.setFont(QtGui.QFont("Arial", 9, QtGui.QFont.Bold))
        painter.drawText(caption, QtCore.Qt.AlignCenter, self.elide(painter, text, caption.width() - 24))

    def paint_background(self, painter: QtGui.QPainter, top="#0f172a", bottom="#020617"):
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        gradient = QtGui.QLinearGradient(0, 0, 0, GAME_HEIGHT)
        gradient.setColorAt(0, QtGui.QColor(top))
        gradient.setColorAt(1, QtGui.QColor(bottom))
        painter.fillRect(self.rect(), gradient)

    def draw_glow_circle(self, painter: QtGui.QPainter, x: float, y: float, radius: float, color: str):
        glow = QtGui.QColor(color)
        glow.setAlpha(45)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(glow)
        painter.drawEllipse(QtCore.QPointF(x, y), radius * 1.9, radius * 1.9)
        painter.setBrush(QtGui.QColor(color))
        painter.drawEllipse(QtCore.QPointF(x, y), radius, radius)


class FlappyBirdGame(GameCanvas):
    title = "Flappy Charger Bird"
    control_hint = "Blink or Space flaps. Dodge every charger tower."
    accent = "#7cff6b"

    def reset(self):
        super().reset()
        self.bird_y = GAME_HEIGHT / 2
        self.velocity = 0.0
        self.pipes = []
        self.tick = 0
        self.alive = True
        self.status = "fly through charger gaps"

    def __init__(self):
        super().__init__()
        self.reset()

    def handle_action(self, action: str):
        if action == "eye_blink":
            self.flap()

    def handle_key(self, key: int):
        super().handle_key(key)
        if key == QtCore.Qt.Key_Space:
            self.flap()

    def flap(self):
        if not self.alive:
            self.reset()
        self.velocity = -7.5

    def step(self):
        if not self.alive:
            return
        self.tick += 1
        self.velocity += 0.45
        self.bird_y += self.velocity
        if self.tick % 85 == 0:
            gap_y = random.randint(120, GAME_HEIGHT - 110)
            self.pipes.append({"x": GAME_WIDTH, "gap_y": gap_y, "scored": False})
        for pipe in self.pipes:
            pipe["x"] -= 3
            if not pipe["scored"] and pipe["x"] < 90:
                pipe["scored"] = True
                self.score += 1
        self.pipes = [pipe for pipe in self.pipes if pipe["x"] > -70]
        if self.bird_y < 55 or self.bird_y > GAME_HEIGHT - 20:
            self.alive = False
            self.status = "crashed - blink or Space to restart"
        for pipe in self.pipes:
            in_x = pipe["x"] < 110 and pipe["x"] + 55 > 70
            in_gap = pipe["gap_y"] - 55 < self.bird_y < pipe["gap_y"] + 55
            if in_x and not in_gap:
                self.alive = False
                self.status = "charger clipped an obstacle - blink or Space to restart"

    def draw_bird(self, painter: QtGui.QPainter):
        x = 92
        y = float(self.bird_y)

        cable = QtGui.QPainterPath()
        cable.moveTo(x - 16, y + 9)
        cable.cubicTo(x - 46, y + 28, x - 66, y - 22, x - 92, y + 4)
        painter.setPen(QtGui.QPen(QtGui.QColor("#f8fafc"), 4, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
        painter.drawPath(cable)

        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor("#e2e8f0"))
        painter.drawRoundedRect(QtCore.QRectF(x - 105, y - 6, 16, 14), 3, 3)
        painter.setBrush(QtGui.QColor("#94a3b8"))
        painter.drawRect(QtCore.QRectF(x - 109, y - 2, 4, 3))
        painter.drawRect(QtCore.QRectF(x - 109, y + 4, 4, 3))

        wing_color = QtGui.QColor("#f59e0b" if self.velocity > 0 else "#fbbf24")
        painter.setBrush(QtGui.QColor("#facc15"))
        painter.drawEllipse(QtCore.QPointF(x, y), 22, 17)
        painter.setBrush(wing_color)
        wing = QtGui.QPolygonF(
            [
                QtCore.QPointF(x - 8, y + 1),
                QtCore.QPointF(x - 30, y + (18 if self.velocity > 0 else -20)),
                QtCore.QPointF(x + 4, y + 10),
            ]
        )
        painter.drawPolygon(wing)

        painter.setBrush(QtGui.QColor("#fb923c"))
        beak = QtGui.QPolygonF(
            [
                QtCore.QPointF(x + 20, y - 3),
                QtCore.QPointF(x + 39, y + 4),
                QtCore.QPointF(x + 19, y + 10),
            ]
        )
        painter.drawPolygon(beak)

        painter.setBrush(QtGui.QColor("#0f172a"))
        painter.drawEllipse(QtCore.QPointF(x + 10, y - 7), 3.6, 3.6)
        painter.setBrush(QtGui.QColor("#38bdf8"))
        painter.drawRoundedRect(QtCore.QRectF(x - 3, y + 12, 18, 11), 3, 3)
        painter.setPen(QtGui.QPen(QtGui.QColor("#bae6fd"), 1.2))
        painter.drawLine(QtCore.QPointF(x + 2, y + 17), QtCore.QPointF(x + 10, y + 17))
        painter.setPen(QtCore.Qt.NoPen)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        self.paint_background(painter, "#07130b", "#020617")
        painter.setPen(QtGui.QPen(QtGui.QColor(124, 255, 107, 38), 1))
        for y in range(88, GAME_HEIGHT, 38):
            painter.drawLine(0, y, GAME_WIDTH, y - 34)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(124, 255, 107, 22))
        painter.drawEllipse(QtCore.QPointF(440, 118), 74, 74)
        self.draw_header(painter)
        for pipe in self.pipes:
            x = int(pipe["x"])
            gap_y = int(pipe["gap_y"])
            tower_gradient = QtGui.QLinearGradient(x, 55, x + 62, 55)
            tower_gradient.setColorAt(0, QtGui.QColor("#15803d"))
            tower_gradient.setColorAt(0.5, QtGui.QColor("#86efac"))
            tower_gradient.setColorAt(1, QtGui.QColor("#166534"))
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(tower_gradient)
            painter.drawRoundedRect(QtCore.QRectF(x, 64, 62, gap_y - 119), 10, 10)
            painter.drawRoundedRect(QtCore.QRectF(x, gap_y + 55, 62, GAME_HEIGHT - gap_y - 88), 10, 10)

            painter.setBrush(QtGui.QColor("#052e16"))
            painter.drawRoundedRect(QtCore.QRectF(x - 6, gap_y - 67, 74, 14), 7, 7)
            painter.drawRoundedRect(QtCore.QRectF(x - 6, gap_y + 53, 74, 14), 7, 7)
            painter.setPen(QtGui.QPen(QtGui.QColor("#bbf7d0"), 2))
            painter.drawLine(x + 13, gap_y - 30, x + 28, gap_y - 48)
            painter.drawLine(x + 28, gap_y - 48, x + 22, gap_y - 22)
            painter.drawLine(x + 39, gap_y + 31, x + 25, gap_y + 48)
            painter.drawLine(x + 25, gap_y + 48, x + 31, gap_y + 21)

        self.draw_bird(painter)
        self.draw_caption(painter, "Goal: blink to fly the charger bird through neon tower gaps.")


class PongGame(GameCanvas):
    title = "Pong"
    control_hint = "Right squeeze/Up moves up. Left squeeze/Down moves down."
    accent = "#7cff6b"

    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        super().reset()
        self.player_y = GAME_HEIGHT / 2
        self.cpu_y = GAME_HEIGHT / 2
        self.ball_x = GAME_WIDTH / 2
        self.ball_y = GAME_HEIGHT / 2
        self.ball_vx = 4.5
        self.ball_vy = 3.2
        self.status = "squeeze to steer the paddle"

    def handle_action(self, action: str):
        if action == "right_squeeze":
            self.player_y -= 28
        elif action == "left_squeeze":
            self.player_y += 28
        self.player_y = max(80, min(GAME_HEIGHT - 40, self.player_y))

    def handle_key(self, key: int):
        super().handle_key(key)
        if key == QtCore.Qt.Key_Up:
            self.handle_action("right_squeeze")
        elif key == QtCore.Qt.Key_Down:
            self.handle_action("left_squeeze")

    def step(self):
        self.cpu_y += max(-3.0, min(3.0, self.ball_y - self.cpu_y))
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy
        if self.ball_y < 60 or self.ball_y > GAME_HEIGHT - 15:
            self.ball_vy *= -1
        if self.ball_x < 45 and abs(self.ball_y - self.player_y) < 48:
            self.ball_vx = abs(self.ball_vx)
            self.score += 1
        if self.ball_x > GAME_WIDTH - 45 and abs(self.ball_y - self.cpu_y) < 48:
            self.ball_vx = -abs(self.ball_vx)
        if self.ball_x < 0:
            self.status = "miss - reset"
            self.reset()
        elif self.ball_x > GAME_WIDTH:
            self.ball_vx = -abs(self.ball_vx)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        self.paint_background(painter, "#08110c", "#020617")
        painter.setPen(QtGui.QPen(QtGui.QColor(124, 255, 107, 28), 1))
        for x in range(0, GAME_WIDTH, 34):
            painter.drawLine(x, 70, x + 90, GAME_HEIGHT)
        self.draw_header(painter)
        painter.setPen(QtGui.QPen(QtGui.QColor("#334155"), 2, QtCore.Qt.DashLine))
        painter.drawLine(GAME_WIDTH // 2, 74, GAME_WIDTH // 2, GAME_HEIGHT - 22)

        player = QtCore.QRectF(24, int(self.player_y - 45), 18, 90)
        cpu = QtCore.QRectF(GAME_WIDTH - 42, int(self.cpu_y - 45), 18, 90)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(96, 165, 250, 60))
        painter.drawRoundedRect(player.adjusted(-7, -7, 7, 7), 12, 12)
        painter.setBrush(QtGui.QColor("#60a5fa"))
        painter.drawRoundedRect(player, 9, 9)
        painter.setBrush(QtGui.QColor(248, 113, 113, 60))
        painter.drawRoundedRect(cpu.adjusted(-7, -7, 7, 7), 12, 12)
        painter.setBrush(QtGui.QColor("#f87171"))
        painter.drawRoundedRect(cpu, 9, 9)
        self.draw_glow_circle(painter, self.ball_x, self.ball_y, 9, "#f8fafc")
        self.draw_caption(painter, "Goal: squeeze to steer the paddle and keep the energy ball alive.")


class CrossyRoadGame(GameCanvas):
    title = "Crossy Road"
    control_hint = "Left squeeze, Space, or Up hops forward."
    accent = "#7cff6b"

    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        super().reset()
        self.player_lane = 0
        self.cars = [
            {"lane": lane, "x": random.randint(0, GAME_WIDTH), "speed": random.choice([-3, -2, 2, 3])}
            for lane in range(1, 7)
        ]
        self.status = "left squeeze or Space moves forward"

    def handle_action(self, action: str):
        if action == "left_squeeze":
            self.player_lane += 1
            if self.player_lane >= 7:
                self.score += 1
                self.player_lane = 0

    def handle_key(self, key: int):
        super().handle_key(key)
        if key in (QtCore.Qt.Key_Space, QtCore.Qt.Key_Up):
            self.handle_action("left_squeeze")

    def step(self):
        lane_height = 48
        player_y = GAME_HEIGHT - 40 - self.player_lane * lane_height
        for car in self.cars:
            car["x"] += car["speed"]
            if car["x"] < -70:
                car["x"] = GAME_WIDTH + 20
            elif car["x"] > GAME_WIDTH + 70:
                car["x"] = -60
            car_y = GAME_HEIGHT - 40 - car["lane"] * lane_height
            if car["lane"] == self.player_lane and abs(car["x"] - GAME_WIDTH / 2) < 35 and abs(car_y - player_y) < 25:
                self.status = "hit - reset"
                self.player_lane = 0

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        self.paint_background(painter, "#14532d", "#052e16")
        self.draw_header(painter)
        lane_height = 48
        for lane in range(1, 7):
            y = GAME_HEIGHT - 72 - lane * lane_height
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QColor(15, 23, 42, 210))
            painter.drawRoundedRect(QtCore.QRectF(10, y - 4, GAME_WIDTH - 20, 36), 8, 8)
            painter.setPen(QtGui.QPen(QtGui.QColor("#64748b"), 1, QtCore.Qt.DashLine))
            painter.drawLine(28, y + 14, GAME_WIDTH - 28, y + 14)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(34, 197, 94, 90))
        painter.drawRoundedRect(QtCore.QRectF(10, GAME_HEIGHT - 76, GAME_WIDTH - 20, 42), 12, 12)
        painter.drawRoundedRect(QtCore.QRectF(10, 72, GAME_WIDTH - 20, 38), 12, 12)

        for car in self.cars:
            y = GAME_HEIGHT - 55 - car["lane"] * lane_height
            x = int(car["x"])
            color = "#f97316" if car["speed"] > 0 else "#38bdf8"
            painter.setBrush(QtGui.QColor(color))
            painter.drawRoundedRect(QtCore.QRectF(x, y, 58, 26), 8, 8)
            painter.setBrush(QtGui.QColor("#0f172a"))
            painter.drawEllipse(QtCore.QPointF(x + 13, y + 25), 4, 4)
            painter.drawEllipse(QtCore.QPointF(x + 45, y + 25), 4, 4)
            painter.setBrush(QtGui.QColor("#fef3c7"))
            painter.drawRect(QtCore.QRectF(x + (48 if car["speed"] > 0 else 5), y + 6, 5, 5))

        player_y = GAME_HEIGHT - 48 - self.player_lane * lane_height
        player_x = int(GAME_WIDTH / 2)
        painter.setBrush(QtGui.QColor(250, 204, 21, 60))
        painter.drawEllipse(QtCore.QPointF(player_x, player_y + 13), 24, 18)
        painter.setBrush(QtGui.QColor("#fde047"))
        painter.drawRoundedRect(QtCore.QRectF(player_x - 15, player_y - 2, 30, 28), 10, 10)
        painter.setBrush(QtGui.QColor("#0f172a"))
        painter.drawEllipse(QtCore.QPointF(player_x - 6, player_y + 8), 2.5, 2.5)
        painter.drawEllipse(QtCore.QPointF(player_x + 6, player_y + 8), 2.5, 2.5)
        self.draw_caption(painter, "Goal: hop lane by lane, avoid traffic, and reach the glowing safe zone.")


class GeometryDashGame(GameCanvas):
    title = "Geometry Dash"
    control_hint = "Right squeeze or Space jumps."
    accent = "#7cff6b"

    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        super().reset()
        self.player_y = GAME_HEIGHT - 70
        self.velocity = 0.0
        self.on_ground = True
        self.obstacles = [{"x": GAME_WIDTH + idx * 190} for idx in range(4)]
        self.status = "right squeeze or Space jumps"

    def handle_action(self, action: str):
        if action == "right_squeeze" and self.on_ground:
            self.velocity = -10.0
            self.on_ground = False

    def handle_key(self, key: int):
        super().handle_key(key)
        if key == QtCore.Qt.Key_Space:
            self.handle_action("right_squeeze")

    def step(self):
        ground = GAME_HEIGHT - 70
        if not self.on_ground:
            self.velocity += 0.55
            self.player_y += self.velocity
            if self.player_y >= ground:
                self.player_y = ground
                self.velocity = 0.0
                self.on_ground = True
        for obstacle in self.obstacles:
            obstacle["x"] -= 5
            if obstacle["x"] < -30:
                obstacle["x"] = GAME_WIDTH + random.randint(120, 240)
                self.score += 1
            if 80 < obstacle["x"] < 122 and self.player_y > ground - 28:
                self.status = "hit - reset"
                self.reset()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        self.paint_background(painter, "#10140a", "#020617")
        painter.setPen(QtGui.QPen(QtGui.QColor(124, 255, 107, 30), 1))
        for x in range(-GAME_HEIGHT, GAME_WIDTH, 34):
            painter.drawLine(x, 80, x + GAME_HEIGHT, GAME_HEIGHT)
        self.draw_header(painter)
        ground = GAME_HEIGHT - 45
        painter.setPen(QtGui.QPen(QtGui.QColor("#7cff6b"), 4))
        painter.drawLine(0, ground, GAME_WIDTH, ground)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(56, 189, 248, 55))
        painter.drawRoundedRect(QtCore.QRectF(82, int(self.player_y) - 6, 38, 38), 8, 8)
        painter.setBrush(QtGui.QColor("#38bdf8"))
        painter.drawRoundedRect(QtCore.QRectF(88, int(self.player_y), 26, 26), 5, 5)
        painter.setBrush(QtGui.QColor("#e0f2fe"))
        painter.drawRect(QtCore.QRectF(96, int(self.player_y) + 7, 4, 4))
        painter.drawRect(QtCore.QRectF(106, int(self.player_y) + 7, 4, 4))
        for obstacle in self.obstacles:
            x = int(obstacle["x"])
            painter.setBrush(QtGui.QColor(251, 113, 133, 60))
            glow = [
                QtCore.QPoint(x - 7, ground),
                QtCore.QPoint(x + 33, ground),
                QtCore.QPoint(x + 13, ground - 43),
            ]
            painter.drawPolygon(QtGui.QPolygon(glow))
            painter.setBrush(QtGui.QColor("#fb7185"))
            points = [
                QtCore.QPoint(x, ground),
                QtCore.QPoint(x + 26, ground),
                QtCore.QPoint(x + 13, ground - 32),
            ]
            painter.drawPolygon(QtGui.QPolygon(points))
        self.draw_caption(painter, "Goal: jump the cube over spikes with right-squeeze impulses.")


def game_visualizer(
    queue: Queue,
    action_queue: Queue | None = None,
    marker_counter=None,
    shutdown_event=None,
    nbChannels=8,
    samplingRate=250,
    apply_filter=True,
    lowcut=15,
    highcut=30.0,
    order=2,
    display_channels=None,
):
    if display_channels is None:
        display_channels = list(range(nbChannels))

    app = QtWidgets.QApplication([])
    main_widget = QtWidgets.QWidget()
    main_widget.setWindowTitle("BCI Inference Games")
    main_widget.resize(1280, 680)
    layout = QtWidgets.QHBoxLayout(main_widget)

    left_panel = QtWidgets.QVBoxLayout()
    sidebar = QtWidgets.QHBoxLayout()
    checkboxes = []
    for ch in range(nbChannels):
        cb = QtWidgets.QCheckBox(CHANNEL_NAMES[ch])
        cb.setChecked(ch in display_channels)
        checkboxes.append(cb)
        sidebar.addWidget(cb)
    left_panel.addLayout(sidebar)

    action_label = QtWidgets.QLabel("BCI action: nothing")
    action_label.setStyleSheet("font-weight: bold; font-size: 16px; color: #7cff6b;")
    left_panel.addWidget(action_label)
    input_label = QtWidgets.QLabel("Last input: none")
    input_label.setStyleSheet("font-size: 14px; color: #e6edf3;")
    left_panel.addWidget(input_label)
    control_label = QtWidgets.QLabel("Mapped control: none")
    control_label.setStyleSheet("font-size: 14px; color: #93c5fd;")
    left_panel.addWidget(control_label)

    plot_widget = pg.GraphicsLayoutWidget()
    left_panel.addWidget(plot_widget, stretch=1)
    plot = plot_widget.addPlot(title="Live EEG Waves")
    plot.showGrid(x=True, y=True)
    plot.setLabel("left", "Amplitude")
    plot.setLabel("bottom", "Time (samples)")
    layout.addLayout(left_panel, stretch=3)

    tabs = QtWidgets.QTabWidget()
    games = [FlappyBirdGame(), PongGame(), CrossyRoadGame(), GeometryDashGame()]
    for game in games:
        tabs.addTab(game, game.title)
    layout.addWidget(tabs, stretch=2)

    main_widget.show()

    buffer_size = samplingRate * 5
    data_buffers = [np.zeros(buffer_size) for _ in range(nbChannels)]
    if apply_filter:
        b, a = butter_bandpass(lowcut, highcut, fs=samplingRate, order=order)
        filter_states = [lfilter_zi(b, a) * 0 for _ in range(nbChannels)]
    else:
        b, a, filter_states = None, None, None

    active_curves = {}
    marker_lines = []

    def active_game():
        return tabs.currentWidget()

    def key_name(key: int) -> str:
        if key == QtCore.Qt.Key_Space:
            return "Space"
        if key == QtCore.Qt.Key_Up:
            return "Up"
        if key == QtCore.Qt.Key_Down:
            return "Down"
        if key == QtCore.Qt.Key_R:
            return "R"
        return QtGui.QKeySequence(key).toString() or str(key)

    def mapped_control(game, action: str) -> str:
        if isinstance(game, FlappyBirdGame) and action in {"eye_blink", "Space"}:
            return "Flappy Bird: flap"
        if isinstance(game, PongGame):
            if action in {"right_squeeze", "Up"}:
                return "Pong: paddle up"
            if action in {"left_squeeze", "Down"}:
                return "Pong: paddle down"
        if isinstance(game, CrossyRoadGame) and action in {"left_squeeze", "Space", "Up"}:
            return "Crossy Road: hop forward"
        if isinstance(game, GeometryDashGame) and action in {"right_squeeze", "Space"}:
            return "Geometry Dash: jump"
        if action == "R":
            return "Reset active game"
        return "No control mapped for this game"

    def add_marker():
        line = pg.InfiniteLine(pos=buffer_size, angle=90, pen=pg.mkPen("r", width=2, style=QtCore.Qt.DashLine))
        plot.addItem(line)
        marker_lines.append(line)
        if marker_counter is not None:
            marker_counter.value += 1

    def handle_manual_key(key: int):
        game = active_game()
        name = key_name(key)
        input_label.setText(f"Last input: keyboard {name}")
        control_label.setText(f"Mapped control: {mapped_control(game, name)}")
        if hasattr(game, "handle_key"):
            game.handle_key(key)
        if key == QtCore.Qt.Key_Space:
            add_marker()

    class KeyFilter(QtCore.QObject):
        def eventFilter(self, obj, event):
            if event.type() == QtCore.QEvent.KeyPress and not event.isAutoRepeat():
                handle_manual_key(event.key())
                return True
            return False

    key_filter = KeyFilter(main_widget)
    main_widget.installEventFilter(key_filter)

    def rebuild_curves():
        enabled = {ch for ch in range(nbChannels) if checkboxes[ch].isChecked()}
        for ch in list(active_curves):
            if ch not in enabled:
                plot.removeItem(active_curves[ch])
                del active_curves[ch]
        for ch in sorted(enabled):
            if ch not in active_curves:
                active_curves[ch] = plot.plot(pen=pg.intColor(ch, nbChannels), name=CHANNEL_NAMES[ch])

    rebuild_curves()
    for cb in checkboxes:
        cb.stateChanged.connect(lambda _: rebuild_curves())

    def drain_actions():
        if action_queue is None:
            return
        while True:
            try:
                event = action_queue.get_nowait()
            except Empty:
                break
            if not isinstance(event, dict):
                continue
            action = event.get("action", "nothing")
            action_label.setText(f"BCI action: {action}")
            game = active_game()
            sample_index = event.get("sample_index", "?")
            input_label.setText(f"Last input: BCI {action} @ sample {sample_index}")
            control_label.setText(f"Mapped control: {mapped_control(game, action)}")
            if hasattr(game, "handle_action"):
                game.handle_action(action)

    def update_waves():
        new_samples = 0
        while True:
            try:
                samples = queue.get_nowait()
            except Empty:
                break
            if samples is None:
                app.quit()
                return
            if not isinstance(samples, np.ndarray) or samples.ndim != 2 or samples.shape[1] != nbChannels:
                continue
            for i in range(samples.shape[0]):
                sample = samples[i, :]
                for ch in range(nbChannels):
                    val = float(sample[ch])
                    if apply_filter:
                        y, filter_states[ch] = lfilter(b, a, [val], zi=filter_states[ch])
                        val = float(y[0])
                    data_buffers[ch] = np.roll(data_buffers[ch], -1)
                    data_buffers[ch][-1] = val
                new_samples += 1

        for line in marker_lines:
            line.setValue(line.value() - new_samples)
        for line in marker_lines[:]:
            if line.value() < 0:
                plot.removeItem(line)
                marker_lines.remove(line)

        for rank, ch in enumerate(sorted(active_curves.keys())):
            active_curves[ch].setData(data_buffers[ch] + 100 * rank)

    def update():
        drain_actions()
        update_waves()
        game = active_game()
        if hasattr(game, "step"):
            game.step()
            game.update()
        if shutdown_event is not None and shutdown_event.is_set():
            app.quit()

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(30)
    app.exec_()
