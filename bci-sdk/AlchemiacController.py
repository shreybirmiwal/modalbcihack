
import sys
import csv
import atexit
import asyncio
import platform
import numpy as np
from time import time, sleep
import signal
from datetime import datetime
from multiprocessing import Process, Queue, freeze_support
from threading import Event
from bleak import BleakScanner

from utils.visualization import visualizer
from proxies.AlchemiacProxy import AlchemiacProxy


freeze_support()

# ─── Platform ────────────────────────────────────────────────────────────────
# On macOS (Darwin), CoreBluetooth exposes UUIDs instead of MAC addresses.
# bleak.BleakScanner returns device.address in the correct format per platform,
# so no manual conversion is needed — just always use device.address.
OS = platform.system()          # "Windows" | "Darwin" | "Linux"
DEVICE_NAME = "Hermes V1"
SCAN_TIMEOUT = 8.0              # seconds
DEVICE_NAME_KEYWORDS = ("hermes", "alchemiac")
DEVICE_MAC_PREFIXES = ("00:80:E1",)
KNOWN_SERVICE_UUID_PREFIXES = ("9fa48",)

# ─── Data files ──────────────────────────────────────────────────────────────
_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
eeg_file    = open(f"data/alchemiac_eeg_{_timestamp}.csv",    "w", newline="")
motion_file = open(f"data/alchemiac_motion_{_timestamp}.csv", "w", newline="")

eeg_writer    = csv.writer(eeg_file)
motion_writer = csv.writer(motion_file)
eeg_writer.writerow(["AF8", "AF7", "CHEEK_R", "CHEEK_L", "EAR_R", "AFz", "BROW_L", "NOSE"])
motion_writer.writerow(["x(g)", "y(g)", "z(g)", "x(deg/s)", "y(deg/s)", "z(deg/s)", "x(G)", "y(G)", "z(G)"])


def _close_files():
    eeg_file.close()
    motion_file.close()

atexit.register(_close_files)


# ─── Callbacks ───────────────────────────────────────────────────────────────
q = None


def eeg_callback(samples):
    global q
    if not samples:           # skip empty packet (e.g. 0-byte data after header)
        return
    arr = np.array(samples, dtype=np.float64)
    # Only forward well-formed 2-D arrays; log anything unexpected
    if arr.ndim != 2 or arr.shape[1] != 8:
        print(f"[EEG] Unexpected sample array shape {arr.shape}, skipping visualizer update.")
    elif q is not None:
        q.put(arr)
    eeg_writer.writerows(samples)


def motion_callback(sample):
    ax, ay, az, gx, gy, gz, cx, cy, cz = sample
    motion_writer.writerow([
        f"{ax:.5f}", f"{ay:.5f}", f"{az:.5f}",
        f"{gx:.5f}", f"{gy:.5f}", f"{gz:.5f}",
        f"{cx:.5f}", f"{cy:.5f}", f"{cz:.5f}",
    ])


# ─── Graceful shutdown ───────────────────────────────────────────────────────
shutdown_event = Event()


def _signal_handler(sig, frame):
    print("\n[INFO] Ctrl+C received. Initiating shutdown...")
    shutdown_event.set()


signal.signal(signal.SIGINT, _signal_handler)


def sleep_until_shutdown(seconds: float):
    end = time() + seconds
    while not shutdown_event.is_set() and time() < end:
        sleep(min(0.1, end - time()))


# ─── BLE scanning & device selection ─────────────────────────────────────────
async def _scan(timeout: float = SCAN_TIMEOUT) -> list:
    """Return likely Alchemiac BLE devices plus a macOS fallback candidate list."""
    addr_kind = "UUID" if OS == "Darwin" else "MAC address"
    print(f"\n[SCAN] Scanning for '{DEVICE_NAME}' ({timeout:.0f} s)  "
          f"[{OS} — using {addr_kind}]\n")
    discovered = await BleakScanner.discover(timeout=timeout, return_adv=True)

    matched = []
    mac_fallback = []

    for device, adv in discovered.values():
        names = [value for value in (adv.local_name, device.name) if value]
        lowered_names = [value.lower() for value in names]
        address = (device.address or "").upper()
        service_uuids = [uuid.lower() for uuid in (adv.service_uuids or [])]

        reasons = []
        if any(DEVICE_NAME.lower() in value for value in lowered_names):
            reasons.append("name")
        elif any(keyword in value for value in lowered_names for keyword in DEVICE_NAME_KEYWORDS):
            reasons.append("name-keyword")

        if any(uuid.startswith(prefix) for uuid in service_uuids for prefix in KNOWN_SERVICE_UUID_PREFIXES):
            reasons.append("service-uuid")

        if any(address.startswith(prefix) for prefix in DEVICE_MAC_PREFIXES):
            reasons.append("mac-prefix")

        if reasons:
            matched.append({
                "device": device,
                "adv": adv,
                "reasons": reasons,
            })
            continue

        if OS == "Darwin" and names:
            mac_fallback.append({
                "device": device,
                "adv": adv,
                "reasons": ["macos-fallback"],
            })

    if matched:
        return matched
    if OS == "Darwin":
        print("[SCAN] No exact Hermes match. On macOS the headset may advertise under a manufacturer name until after the first connection.")
        return mac_fallback
    return []


def _select_device(devices: list):
    """Print a numbered menu and return the chosen BleakDevice."""
    addr_label = "UUID" if OS == "Darwin" else "Address"
    width = 86
    print("─" * width)
    print(f"  Found {len(devices)} candidate device(s):")
    print("─" * width)
    for i, candidate in enumerate(devices):
        dev = candidate["device"]
        adv = candidate["adv"]
        name = adv.local_name or dev.name or "<unknown>"
        if adv.local_name and dev.name and adv.local_name != dev.name:
            name = f"{adv.local_name} / {dev.name}"
        reasons = ", ".join(candidate["reasons"])
        print(f"  [{i}]  {name:<32}  {addr_label}: {dev.address}  [{reasons}]")
    print("─" * width)

    while True:
        try:
            raw = input(f"\n  Select device [0–{len(devices) - 1}]: ").strip()
            idx = int(raw)
            if 0 <= idx < len(devices):
                return devices[idx]["device"]
            print(f"  Enter a number between 0 and {len(devices) - 1}.")
        except ValueError:
            print("  Invalid input — enter a number.")


def scan_and_select() -> str:
    """
    Scan for Hermes devices, loop until at least one is found,
    let the user choose, and return the address ready for BleakClient.
    """
    while True:
        devices = asyncio.run(_scan())
        if devices:
            break
        print(f"[SCAN] No '{DEVICE_NAME}' candidates found.")
        print(f"[SCAN] Expected non-macOS MAC prefix: {DEVICE_MAC_PREFIXES[0]}:XX:XX:XX")
        ans = input("  Press Enter to scan again, or 'q' to quit: ").strip().lower()
        if ans == "q":
            print("[INFO] Exiting.")
            sys.exit(0)

    chosen = _select_device(devices)
    print(f"\n[INFO] Selected: {chosen.name}  ({chosen.address})\n")
    return chosen.address


# ─── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import multiprocessing
    # "spawn" is already the default on macOS (Python ≥ 3.8) and Windows.
    # Forced here so behaviour is identical on all platforms.
    multiprocessing.set_start_method("spawn", force=True)

    # 1. Discover and select the device
    device_address = scan_and_select()

    # 2. Start the EEG visualizer in a separate process
    q = Queue()
    vis_process = Process(target=visualizer, args=(q,))
    vis_process.start()

    proxy = None
    try:
        # 3. Connect
        proxy = AlchemiacProxy(device_address,
                               eeg_callback=eeg_callback,
                               motion_callback=motion_callback)
        proxy.waitForConnected()

        # Brief stabilisation pause before the live demo
        print("Stabilising signal…")
        sleep_until_shutdown(1)
        sleep_until_shutdown(10)

        # 4. Stream for the demo duration
        if not shutdown_event.is_set():
            print("Starting experience")
            sleep_until_shutdown(240)

    except KeyboardInterrupt:
        print("[MAIN] KeyboardInterrupt caught.")

    finally:
        print("[MAIN] Cleaning up…")
        shutdown_event.set()
        if proxy is not None:
            proxy.disconnect()
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
        print("[MAIN] Shutdown complete. Goodbye!")
