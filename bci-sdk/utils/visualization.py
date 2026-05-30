from multiprocessing import Queue
from queue import Empty
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from scipy.signal import butter, lfilter, lfilter_zi


CHANNEL_NAMES = ["AF8", "AF7", "CHEEK_R", "CHEEK_L", "EAR_R", "AFz", "BROW_L", "NOSE"]


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    return butter(order, [low, high], btype='band')


def visualizer(queue: Queue, marker_queue=None, marker_flag=None, shutdown_event=None,
               nbChannels=8, samplingRate=250,
               apply_filter=True, lowcut=15, highcut=30.0, order=2,
               display_channels=None):

    if display_channels is None:
        display_channels = list(range(nbChannels))

    app = QtWidgets.QApplication([])

    main_widget = QtWidgets.QWidget()
    main_widget.setWindowTitle("Real-time EEG")
    main_widget.resize(1000, 500)
    layout = QtWidgets.QHBoxLayout(main_widget)

    # Sidebar
    sidebar = QtWidgets.QVBoxLayout()
    sidebar_label = QtWidgets.QLabel("Channels")
    sidebar_label.setStyleSheet("font-weight: bold; font-size: 14px;")
    sidebar.addWidget(sidebar_label)

    checkboxes = []
    for ch in range(nbChannels):
        cb = QtWidgets.QCheckBox(CHANNEL_NAMES[ch])
        cb.setChecked(ch in display_channels)
        checkboxes.append(cb)
        sidebar.addWidget(cb)

    sidebar.addStretch()
    sidebar.addWidget(QtWidgets.QLabel("Press SPACE to mark"))
    layout.addLayout(sidebar)

    # Plot
    plot_widget = pg.GraphicsLayoutWidget()
    layout.addWidget(plot_widget, stretch=1)
    plot = plot_widget.addPlot(title="EEG Channels")
    plot.showGrid(x=True, y=True)
    plot.setLabel('left', 'Amplitude (uV)')
    plot.setLabel('bottom', 'Time (samples)')

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

    def add_marker():
        line = pg.InfiniteLine(pos=buffer_size, angle=90,
                               pen=pg.mkPen('r', width=2, style=QtCore.Qt.DashLine))
        plot.addItem(line)
        marker_lines.append(line)
        if marker_queue is not None:
            marker_queue.put(1)
        if marker_flag is not None:
            marker_flag.value = 1
        print("[MARKER]", flush=True)

    class KeyFilter(QtCore.QObject):
        def eventFilter(self, obj, event):
            if (event.type() == QtCore.QEvent.KeyPress
                    and event.key() == QtCore.Qt.Key_Space
                    and not event.isAutoRepeat()):
                add_marker()
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

    def update():
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

        enabled_sorted = sorted(active_curves.keys())
        for rank, ch in enumerate(enabled_sorted):
            active_curves[ch].setData(data_buffers[ch] + 100 * rank)

        if shutdown_event is not None and shutdown_event.is_set():
            app.quit()

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(30)
    app.exec_()
