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
        painter.setPen(QtGui.QColor("#e6edf3"))
        painter.setFont(QtGui.QFont("Arial", 14, QtGui.QFont.Bold))
        painter.drawText(16, 26, f"{self.title} | score {self.score} | {self.status}")
        painter.setFont(QtGui.QFont("Arial", 10))
        painter.drawText(16, 44, "Keyboard fallback: Space/arrow keys. Press R to reset.")

    def paint_background(self, painter: QtGui.QPainter, color="#101820"):
        painter.fillRect(self.rect(), QtGui.QColor(color))


class FlappyBirdGame(GameCanvas):
    title = "Flappy Bird"

    def reset(self):
        super().reset()
        self.bird_y = GAME_HEIGHT / 2
        self.velocity = 0.0
        self.pipes = []
        self.tick = 0
        self.alive = True
        self.status = "blink or Space to flap"

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
                self.status = "pipe hit - blink or Space to restart"

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        self.paint_background(painter, "#102033")
        self.draw_header(painter)
        painter.setBrush(QtGui.QColor("#ffd166"))
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawEllipse(QtCore.QPointF(90, self.bird_y), 14, 14)
        painter.setBrush(QtGui.QColor("#06d6a0"))
        for pipe in self.pipes:
            x = int(pipe["x"])
            gap_y = int(pipe["gap_y"])
            painter.drawRect(x, 55, 55, gap_y - 110)
            painter.drawRect(x, gap_y + 55, 55, GAME_HEIGHT - gap_y - 55)


class PongGame(GameCanvas):
    title = "Pong"

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
        self.status = "right=up, left=down"

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
        self.paint_background(painter, "#111827")
        self.draw_header(painter)
        painter.setBrush(QtGui.QColor("#60a5fa"))
        painter.drawRect(22, int(self.player_y - 42), 16, 84)
        painter.setBrush(QtGui.QColor("#f87171"))
        painter.drawRect(GAME_WIDTH - 38, int(self.cpu_y - 42), 16, 84)
        painter.setBrush(QtGui.QColor("#f8fafc"))
        painter.drawEllipse(QtCore.QPointF(self.ball_x, self.ball_y), 8, 8)


class CrossyRoadGame(GameCanvas):
    title = "Crossy Road"

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
        self.paint_background(painter, "#17351f")
        self.draw_header(painter)
        lane_height = 48
        painter.setPen(QtGui.QColor("#335c45"))
        for lane in range(8):
            y = GAME_HEIGHT - 64 - lane * lane_height
            painter.drawLine(0, y, GAME_WIDTH, y)
        painter.setBrush(QtGui.QColor("#f97316"))
        for car in self.cars:
            y = GAME_HEIGHT - 55 - car["lane"] * lane_height
            painter.drawRect(int(car["x"]), y, 54, 26)
        painter.setBrush(QtGui.QColor("#fde047"))
        player_y = GAME_HEIGHT - 48 - self.player_lane * lane_height
        painter.drawRect(int(GAME_WIDTH / 2 - 12), player_y, 24, 24)


class GeometryDashGame(GameCanvas):
    title = "Geometry Dash"

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
        self.paint_background(painter, "#1f1535")
        self.draw_header(painter)
        ground = GAME_HEIGHT - 45
        painter.setPen(QtGui.QColor("#c084fc"))
        painter.drawLine(0, ground, GAME_WIDTH, ground)
        painter.setBrush(QtGui.QColor("#38bdf8"))
        painter.drawRect(88, int(self.player_y), 26, 26)
        painter.setBrush(QtGui.QColor("#fb7185"))
        for obstacle in self.obstacles:
            x = int(obstacle["x"])
            points = [
                QtCore.QPoint(x, ground),
                QtCore.QPoint(x + 26, ground),
                QtCore.QPoint(x + 13, ground - 32),
            ]
            painter.drawPolygon(QtGui.QPolygon(points))


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
    action_label.setStyleSheet("font-weight: bold; font-size: 16px;")
    left_panel.addWidget(action_label)

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

    def add_marker():
        line = pg.InfiniteLine(pos=buffer_size, angle=90, pen=pg.mkPen("r", width=2, style=QtCore.Qt.DashLine))
        plot.addItem(line)
        marker_lines.append(line)
        if marker_counter is not None:
            marker_counter.value += 1

    def handle_manual_key(key: int):
        game = active_game()
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
