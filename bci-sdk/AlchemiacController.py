import csv
import numpy as np
import signal
from datetime import datetime
from multiprocessing import Manager, Process, Queue, freeze_support
from pathlib import Path
from threading import Event

from pylsl import StreamInlet, resolve_byprop
from utils.visualization import visualizer


freeze_support()

LSL_STREAM_NAME = "Reak_EEG"
LSL_RESOLVE_TIMEOUT = 10.0
EEG_CHANNELS = ["AF8", "AF7", "CHEEK_R", "CHEEK_L", "EAR_R", "AFz", "BROW_L", "NOSE"]
DATA_DIR = Path("data")


# ─── Graceful shutdown ───────────────────────────────────────────────────────
shutdown_event = Event()


def _signal_handler(sig, frame):
    if not shutdown_event.is_set():
        print("\n[INFO] Ctrl+C received. Initiating shutdown...")
    shutdown_event.set()


signal.signal(signal.SIGINT, _signal_handler)


def _resolve_eeg_stream():
    print(f"[LSL] Resolving EEG stream '{LSL_STREAM_NAME}'...")
    streams = resolve_byprop("name", LSL_STREAM_NAME, minimum=1, timeout=LSL_RESOLVE_TIMEOUT)
    if not streams:
        raise RuntimeError(
            f"No LSL stream named '{LSL_STREAM_NAME}' found. "
            "Start it first with: uv run AlchemiacStreamLSL.py"
        )
    print(f"[LSL] Connected to stream: {streams[0].name()} ({streams[0].type()})")
    return StreamInlet(streams[0], max_buflen=60)


def _marker_delta(marker_counter, last_marker_count):
    marker_count = int(marker_counter.value)
    return marker_count - last_marker_count, marker_count


def record_lsl_eeg(queue, marker_counter):
    DATA_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    eeg_path = DATA_DIR / f"alchemiac_eeg_{timestamp}.csv"
    inlet = _resolve_eeg_stream()

    first_lsl_timestamp = None
    last_marker_count = 0
    pending_markers = 0
    rows_written = 0

    with eeg_path.open("w", newline="") as eeg_file:
        writer = csv.writer(eeg_file)
        writer.writerow(["timestamp", *EEG_CHANNELS, "marker"])
        print(f"[DATA] Writing LSL EEG CSV: {eeg_path.resolve()}")
        print("[INFO] Press SPACE in the visualizer to mark. Press Ctrl+C to stop.")

        while not shutdown_event.is_set():
            samples, lsl_timestamps = inlet.pull_chunk(timeout=0.1, max_samples=256)
            if not samples:
                continue

            arr = np.array(samples, dtype=np.float64)
            if arr.ndim != 2 or arr.shape[1] != len(EEG_CHANNELS):
                print(f"[LSL] Unexpected EEG chunk shape {arr.shape}, skipping.")
                continue

            queue.put(arr)
            new_markers, last_marker_count = _marker_delta(marker_counter, last_marker_count)
            if new_markers > 0:
                pending_markers += new_markers
                print(f"[CSV MARKER] {new_markers} marker(s) queued", flush=True)

            for sample, lsl_timestamp in zip(samples, lsl_timestamps):
                if first_lsl_timestamp is None:
                    first_lsl_timestamp = lsl_timestamp
                marker = 1 if pending_markers > 0 else 0
                if marker:
                    pending_markers -= 1
                writer.writerow([f"{lsl_timestamp - first_lsl_timestamp:.4f}", *sample, marker])
                rows_written += 1

            eeg_file.flush()

    print(f"\n[DATA] {eeg_path.resolve()}  ({rows_written} samples)")


# ─── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import multiprocessing
    # "spawn" is already the default on macOS (Python ≥ 3.8) and Windows.
    # Forced here so behaviour is identical on all platforms.
    multiprocessing.set_start_method("spawn", force=True)

    q = Queue()
    marker_manager = Manager()
    marker_counter = marker_manager.Value('i', 0)
    vis_process = Process(
        target=visualizer,
        args=(q,),
        kwargs={"marker_counter": marker_counter, "display_channels": list(range(len(EEG_CHANNELS)))},
    )
    vis_process.start()

    try:
        record_lsl_eeg(q, marker_counter)

    except KeyboardInterrupt:
        print("[MAIN] KeyboardInterrupt caught.")
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")

    finally:
        print("[MAIN] Cleaning up…")
        shutdown_event.set()
        if q is not None:
            q.put(None)
        if vis_process.is_alive():
            vis_process.join(timeout=5)
        if vis_process.is_alive():
            print("[MAIN] Visualizer did not exit cleanly; terminating...")
            vis_process.terminate()
            vis_process.join()
        vis_process.close()
        if q is not None:
            q.close()
            q.join_thread()
        marker_manager.shutdown()
        print("[MAIN] Shutdown complete. Goodbye!")
