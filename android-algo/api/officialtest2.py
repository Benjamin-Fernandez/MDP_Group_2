import serial
import requests
import re
import time
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Direction mapping (API expects int: 0=NORTH, 1=EAST, 2=SOUTH, 3=WEST)
DIRECTION_MAP = {"NORTH": 0, "EAST": 1, "SOUTH": 2, "WEST": 3}

def sanitize_line(line: str) -> str:
    """Fix cases where OBSTACLE is glued to direction word."""
    line = re.sub(r"(NORTH)(OBSTACLE)", r"\1 \2", line)
    line = re.sub(r"(SOUTH)(OBSTACLE)", r"\1 \2", line)
    line = re.sub(r"(EAST)(OBSTACLE)",  r"\1 \2", line)
    line = re.sub(r"(WEST)(OBSTACLE)",  r"\1 \2", line)
    line = line.replace("CLEARROBOT", "ROBOT")
    return line

def build_payload(messages):
    """Build JSON payload in correct API format:
       - Robot position: multiples of 5 → scale by /5
       - Obstacles: cm → scale by /10
    """
    payload = {"retrying": False, "obstacles": []}

    for msg in messages:
        msg = sanitize_line(msg)
        segments = msg.split()  # split if ROBOT + OBSTACLE are stuck together

        for seg in segments:
            tokens = seg.split(",")
            if tokens[0] == "ROBOT":
                payload["robot_x"] = int(int(tokens[1]))
                payload["robot_y"] = int(int(tokens[2]))
                payload["robot_dir"] = DIRECTION_MAP[tokens[3]]
            elif tokens[0] == "OBSTACLE":
                ob = {
                    "id": int(tokens[1]),
                    "x": int(int(tokens[2]) / 10),
                    "y": int(int(tokens[3]) / 10),
                    "d": DIRECTION_MAP[tokens[4]]
                }
                payload["obstacles"].append(ob)

    return payload

def save_visualization(payload, data, outfile="latest_path.png"):
    """Draw robot, obstacles, path and save to PNG."""
    fig, ax = plt.subplots(figsize=(7,7))
    ax.set_xlim(-0.5, 19.5)
    ax.set_ylim(-0.5, 19.5)
    ax.set_xticks(range(20))
    ax.set_yticks(range(20))
    ax.grid(True, which="both")

    # Obstacles
    for ob in payload.get("obstacles", []):
        ax.add_patch(plt.Rectangle((ob["x"]-0.5, ob["y"]-0.5), 1, 1, color="red"))
        ax.text(ob["x"], ob["y"], str(ob["id"]), color="white",
                ha="center", va="center", fontsize=8, weight="bold")

    # Path
    path = data.get("data", {}).get("path", [])
    if path:
        xs, ys = [p["x"] for p in path], [p["y"] for p in path]
        ax.plot(xs, ys, color="blue", linewidth=2, marker="o", markersize=4)
    else:
        ax.text(10, 10, "NO PATH", color="gray", ha="center", va="center", fontsize=14, weight="bold")

    # Robot
    if "robot_x" in payload:
        rx, ry, rdir = payload["robot_x"], payload["robot_y"], payload["robot_dir"]
        ax.add_patch(plt.Rectangle((rx-0.5, ry-0.5), 2, 2, color="green", alpha=0.3, edgecolor="black"))
        ax.text(rx+0.5, ry+0.5, "R", color="black", ha="center", va="center", fontsize=12, weight="bold")

        dir_map = {0:(0,1), 1:(1,0), 2:(0,-1), 3:(-1,0)}
        if rdir in dir_map:
            dx, dy = dir_map[rdir]
            ax.arrow(rx+0.5, ry+0.5, dx*0.5, dy*0.5,
                     head_width=0.3, head_length=0.3, fc="green", ec="green")

    ax.set_aspect("equal", adjustable="box")
    plt.title("Pathfinding Emulator")
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"📷 Emulator image saved to {outfile}")

def translate_command(api_cmd: str):
    """
    Translate API motor commands into STM32 simple commands.
    Supports: F, B, L, R, FL, FR, BL, BR, S.
    """
    c = api_cmd.strip().upper()

    # Stop-like commands
    if c == "FIN" or c.startswith("SNAP") or c.startswith("S"):
        return "S"

    parts = c.split("|")
    if len(parts) >= 3:
        try:
            left = float(parts[1])
            right = float(parts[2])

            # Pivot if one motor is 0
            if left == 0 and right > 0:
                return "L"
            if right == 0 and left > 0:
                return "R"
            if left == 0 and right < 0:
                return "BL"
            if right == 0 and left < 0:
                return "BR"

            # Forward arcs
            if left > 0 and right > 0:
                if abs(left - right) < 5:
                    return "F"
                elif left > right:
                    return "FR"
                else:
                    return "FL"

            # Backward arcs
            if left < 0 and right < 0:
                if abs(left - right) < 5:
                    return "B"
                elif left < right:
                    return "BL"
                else:
                    return "BR"

            # Spins (opposite signs)
            if left > 0 and right < 0:
                return "L"
            if left < 0 and right > 0:
                return "R"

        except ValueError:
            pass

    return None

def send_to_api(api_url: str, payload: dict, ser):
    print("\n📤 Parsed JSON payload:", payload)
    if "robot_x" not in payload or not payload.get("obstacles"):
        print("⚠️ No robot or no obstacles, skipping send.")
        return None

    try:
        response = requests.post(f"{api_url}/path", json=payload, timeout=100)
        response.raise_for_status()
        data = response.json()
        print("✅ API Response received")

        # Save emulator PNG + raw JSON
        save_visualization(payload, data, "latest_path.png")
        with open("latest_response.json", "w") as f:
            json.dump(data, f, indent=2)

        # Path (debug print)
        path = data.get("data", {}).get("path", [])
        if path:
            path_cells = [(p["x"], p["y"]) for p in path]
            print("📍 Path cells:", path_cells)
        else:
            print("⚠️ No path returned")

        # Commands → translate → send to STM32
        commands = data.get("data", {}).get("commands", [])
        if commands:
            print("🤖 Commands (API):", commands)
            sent_count = 0
            for cmd in commands:
                simple = translate_command(cmd)
                if simple:
                    ser.write((simple + "\n").encode("utf-8"))
                    print(f"➡️ Sent to STM32: {simple}  (from {cmd})")
                    sent_count += 1
                    time.sleep(1.0)
                else:
                    print(f"⚠️ Skipped API cmd (no mapping): {cmd}")
            print(f"📡 Translated {len(commands)} API cmds → Sent {sent_count} simple cmds to STM32")
        else:
            print("⚠️ No commands returned")

        return data

    except requests.RequestException as e:
        print("❌ Error posting to API:", e)
        return None


if __name__ == "__main__":
    api_url = "http://127.0.0.1:5000"

    try:
        ser = serial.Serial("/dev/rfcomm0", 9600, timeout=1)
        print("🔌 Listening on /dev/rfcomm0 ...")
    except serial.SerialException as e:
        print("❌ Could not open /dev/rfcomm0:", e)
        exit(1)

    buffer = ""
    current_messages = []
    last_received = time.time()

    while True:
        chunk = ser.read(ser.in_waiting or 1).decode("utf-8", errors="ignore")
        if chunk:
            buffer += chunk
            if "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if line:
                    print("📩 Received Bluetooth message:", line)

                    if line.startswith("ROBOT"):
                        if current_messages:
                            payload = build_payload(current_messages)
                            send_to_api(api_url, payload, ser)
                        current_messages = [line]
                    else:
                        current_messages.append(line)

                    last_received = time.time()

        if current_messages and (time.time() - last_received > 2):
            payload = build_payload(current_messages)
            send_to_api(api_url, payload, ser)
            current_messages = []

