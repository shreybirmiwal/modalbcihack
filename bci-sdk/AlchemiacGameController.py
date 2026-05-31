import argparse
import csv
from datetime import datetime
from multiprocessing import Event, Manager, Process, Queue, freeze_support
from pathlib import Path
import signal
import time

import numpy as np
from pylsl import StreamInlet, resolve_byprop

from utils.game_visualization import game_visualizer
from utils.live_inference import BCIActionInference, CHANNEL_NAMES, DEFAULT_MODEL_PATH


LSL_STREAM_NAME = "Reak_EEG"
LSL_RESOLVE_TIMEOUT = 10.0
DATA_DIR = Path("data")

shutdown_event = None


def _signal_handler(sig, frame):
    if shutdown_event is not None and not shutdown_event.is_set():
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


def _publish_inference(inference: BCIActionInference, samples: np.ndarray, action_queue: Queue):
    for event in inference.update(samples):
        payload = event.as_dict()
        action_queue.put(payload)
        print(f"[ACTION] {payload['action']} @ sample={payload['sample_index']}", flush=True)


def run_live(queue: Queue, action_queue: Queue, marker_counter, model_path: Path, chunk_size: int):
    DATA_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    eeg_path = DATA_DIR / f"alchemiac_game_eeg_{timestamp}.csv"
    inlet = _resolve_eeg_stream()
    inference = BCIActionInference(model_path=model_path)

    first_lsl_timestamp = None
    last_marker_count = 0
    pending_markers = 0
    rows_written = 0

    with eeg_path.open("w", newline="") as eeg_file:
        writer = csv.writer(eeg_file)
        writer.writerow(["timestamp", *CHANNEL_NAMES, "marker"])
        print(f"[MODEL] {model_path}")
        print(f"[DATA] Writing live game EEG CSV: {eeg_path.resolve()}")
        print("[INFO] Use game tabs on the right. BCI actions drive the active game.")

        while not shutdown_event.is_set():
            samples, lsl_timestamps = inlet.pull_chunk(timeout=0.1, max_samples=chunk_size)
            if not samples:
                continue

            arr = np.array(samples, dtype=np.float64)
            if arr.ndim != 2 or arr.shape[1] != len(CHANNEL_NAMES):
                print(f"[LSL] Unexpected EEG chunk shape {arr.shape}, skipping.")
                continue

            queue.put(arr)
            _publish_inference(inference, arr, action_queue)

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


def run_replay(queue: Queue, action_queue: Queue, model_path: Path, replay_csv: Path, chunk_size: int, sampling_rate: int):
    inference = BCIActionInference(model_path=model_path)
    print(f"[MODEL] {model_path}")
    print(f"[REPLAY] {replay_csv}")
    print("[INFO] Replaying CSV through the same live inference and game path.")

    with replay_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        chunk = []
        for row in reader:
            if shutdown_event.is_set():
                break
            try:
                chunk.append([float(row[name]) for name in CHANNEL_NAMES])
            except (KeyError, ValueError) as exc:
                raise ValueError(f"invalid replay row in {replay_csv}") from exc
            if len(chunk) >= chunk_size:
                arr = np.array(chunk, dtype=np.float64)
                queue.put(arr)
                _publish_inference(inference, arr, action_queue)
                chunk = []
                time.sleep(len(arr) / sampling_rate)

        if chunk and not shutdown_event.is_set():
            arr = np.array(chunk, dtype=np.float64)
            queue.put(arr)
            _publish_inference(inference, arr, action_queue)

    print("[REPLAY] complete")
    while not shutdown_event.is_set():
        time.sleep(0.1)


def parse_args():
    parser = argparse.ArgumentParser(description="Run live BCI model inference with wave plots and game demos.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Path to auto research/runs/final_model.json")
    parser.add_argument("--replay-csv", default="", help="Optional CSV to replay instead of reading LSL.")
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--sampling-rate", type=int, default=250)
    return parser.parse_args()


if __name__ == "__main__":
    import multiprocessing

    freeze_support()
    multiprocessing.set_start_method("spawn", force=True)
    shutdown_event = Event()
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve()
    replay_csv = Path(args.replay_csv).expanduser().resolve() if args.replay_csv else None

    q = Queue()
    action_q = Queue()
    marker_manager = Manager()
    marker_counter = marker_manager.Value("i", 0)
    vis_process = Process(
        target=game_visualizer,
        args=(q,),
        kwargs={
            "action_queue": action_q,
            "marker_counter": marker_counter,
            "shutdown_event": shutdown_event,
            "samplingRate": args.sampling_rate,
            "display_channels": list(range(len(CHANNEL_NAMES))),
        },
    )
    vis_process.start()

    try:
        if replay_csv:
            run_replay(q, action_q, model_path, replay_csv, args.chunk_size, args.sampling_rate)
        else:
            run_live(q, action_q, marker_counter, model_path, args.chunk_size)
    except KeyboardInterrupt:
        print("[MAIN] KeyboardInterrupt caught.")
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
    finally:
        print("[MAIN] Cleaning up...")
        shutdown_event.set()
        q.put(None)
        if vis_process.is_alive():
            vis_process.join(timeout=5)
        if vis_process.is_alive():
            print("[MAIN] Visualizer did not exit cleanly; terminating...")
            vis_process.terminate()
            vis_process.join()
        vis_process.close()
        q.close()
        q.join_thread()
        action_q.close()
        action_q.join_thread()
        marker_manager.shutdown()
        print("[MAIN] Shutdown complete. Goodbye!")
