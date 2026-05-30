
import sys
import asyncio
import platform
import signal
from time import sleep
from multiprocessing import freeze_support

from bleak import BleakScanner
from pylsl import StreamInfo, StreamOutlet

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

# ─── LSL stream configuration ────────────────────────────────────────────────
EEG_CHANNELS = ["AF8", "AF7", "CHEEK_R", "CHEEK_L", "EAR_R", "AFz", "BROW_L", "NOSE"]
EEG_SAMPLING_RATE = 250

MOTION_CHANNELS = ["ax(g)", "ay(g)", "az(g)", "gx(deg/s)", "gy(deg/s)", "gz(deg/s)", "cx(G)", "cy(G)", "cz(G)"]
MOTION_SAMPLING_RATE = 100


# ─── Graceful shutdown ───────────────────────────────────────────────────────
shutdown_flag = False


def _signal_handler(sig, frame):
    global shutdown_flag
    print("\n[INFO] Ctrl+C received. Initiating shutdown...")
    shutdown_flag = True


signal.signal(signal.SIGINT, _signal_handler)


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
    multiprocessing.set_start_method("spawn", force=True)

    # 1. Discover and select the device
    device_address = scan_and_select()

    # 2. Set up LSL outlets
    lsl_eeg_info = StreamInfo(
        'Reak_EEG', 'EEG', len(EEG_CHANNELS), EEG_SAMPLING_RATE, 'float32', 'reak_eeg_001'
    )
    for label in EEG_CHANNELS:
        lsl_eeg_info.desc().append_child("channels").append_child("channel").append_child_value("label", label)
    lsl_eeg_outlet = StreamOutlet(lsl_eeg_info)

    lsl_motion_info = StreamInfo(
        'Reak_Motion', 'Motion', len(MOTION_CHANNELS), MOTION_SAMPLING_RATE, 'float32', 'reak_motion_001'
    )
    for label in MOTION_CHANNELS:
        lsl_motion_info.desc().append_child("channels").append_child("channel").append_child_value("label", label)
    lsl_motion_outlet = StreamOutlet(lsl_motion_info)

    print("[INFO] LSL streams initialized: Reak_EEG + Reak_Motion")

    # 3. Callbacks — forward decoded data directly to LSL outlets
    def eeg_callback(samples):
        for sample in samples:
            lsl_eeg_outlet.push_sample(sample)

    def motion_callback(sample):
        lsl_motion_outlet.push_sample(list(sample))

    proxy = None
    try:
        # 4. Connect
        proxy = AlchemiacProxy(device_address, eeg_callback=eeg_callback, motion_callback=motion_callback)
        proxy.waitForConnected()

        print("Stabilising signal…")
        sleep(1)

        print("[INFO] Streaming EEG + Motion via LSL. Press Ctrl+C to stop.")

        # 5. Run indefinitely until Ctrl+C
        while not shutdown_flag:
            sleep(0.1)

    except KeyboardInterrupt:
        print("[MAIN] KeyboardInterrupt caught.")

    finally:
        print("[MAIN] Cleaning up…")
        if proxy is not None:
            proxy.disconnect()
        print("[MAIN] Shutdown complete. Goodbye!")
