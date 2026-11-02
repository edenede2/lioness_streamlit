#!/usr/bin/env python3
"""Capture actual localhost Streamlit views with Chrome DevTools (stdlib only)."""
from pathlib import Path
import argparse, base64, json, os, socket, struct, time, urllib.request, urllib.parse


class DevTools:
    def __init__(self, url):
        parsed = urllib.parse.urlsplit(url)
        self.sock = socket.create_connection((parsed.hostname, parsed.port), timeout=30)
        key = base64.b64encode(os.urandom(16)).decode()
        request = f"GET {parsed.path} HTTP/1.1\r\nHost: {parsed.netloc}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\nOrigin: http://127.0.0.1:9226\r\n\r\n"
        self.sock.sendall(request.encode())
        header = b""
        while not header.endswith(b"\r\n\r\n"):
            header += self.sock.recv(1)
        if b" 101 " not in header.split(b"\r\n", 1)[0]:
            raise RuntimeError(
                "Chrome rejected local DevTools websocket: "
                + header.split(b"\r\n", 1)[0].decode()
            )
        self.counter = 0

    def receive_bytes(self, n):
        pieces = []
        while n:
            part = self.sock.recv(n)
            if not part:
                raise EOFError("Chrome DevTools socket closed")
            pieces.append(part)
            n -= len(part)
        return b"".join(pieces)

    def send(self, data, opcode=1):
        data = data.encode() if isinstance(data, str) else data
        length = len(data)
        header = bytes([128 | opcode])
        header += (
            bytes([128 | length])
            if length < 126
            else (
                bytes([128 | 126]) + struct.pack("!H", length)
                if length < 65536
                else bytes([128 | 127]) + struct.pack("!Q", length)
            )
        )
        mask = os.urandom(4)
        self.sock.sendall(
            header + mask + bytes(v ^ mask[i % 4] for i, v in enumerate(data))
        )

    def receive(self):
        fragments = []
        while True:
            a, b = self.receive_bytes(2)
            length = b & 127
            if length == 126:
                length = struct.unpack("!H", self.receive_bytes(2))[0]
            elif length == 127:
                length = struct.unpack("!Q", self.receive_bytes(8))[0]
            mask = self.receive_bytes(4) if b & 128 else None
            data = self.receive_bytes(length)
            if mask:
                data = bytes(v ^ mask[i % 4] for i, v in enumerate(data))
            if a & 15 == 9:
                self.send(data, 10)
                continue
            if a & 15 == 8:
                raise EOFError("Chrome closed DevTools")
            fragments.append(data)
            if a & 128:
                return json.loads(b"".join(fragments))

    def call(self, method, params=None):
        self.counter += 1
        identifier = self.counter
        self.send(json.dumps(dict(id=identifier, method=method, params=params or {})))
        while True:
            answer = self.receive()
            if answer.get("id") == identifier:
                if "error" in answer:
                    raise RuntimeError(answer["error"])
                return answer.get("result", {})

    def js(self, expression):
        return self.call(
            "Runtime.evaluate", dict(expression=expression, returnByValue=True)
        )["result"].get("value")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/control_calibrated_implementation_20261102/screenshots"
        ),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    pages = json.load(urllib.request.urlopen("http://127.0.0.1:9226/json/list"))
    page = next(p for p in pages if p["type"] == "page")
    browser = DevTools(page["webSocketDebuggerUrl"])
    browser.call("Page.enable")
    browser.call("Runtime.enable")
    browser.call(
        "Emulation.setDeviceMetricsOverride",
        dict(width=1920, height=1800, deviceScaleFactor=1, mobile=False),
    )
    proof = []
    cases = [
        (
            "brain_gain_loss",
            dict(
                view="Associations",
                screen_partition="full_cohort",
                screen_module="188",
                score_source="calibrated_beta3",
            ),
        ),
        (
            "motor_beta1",
            dict(
                view="Associations",
                screen_partition="muscle_control_l4",
                screen_module="312",
                score_source="calibrated_beta1",
            ),
        ),
        (
            "brain_heatmap",
            dict(
                view="Correlation heatmaps",
                screen_partition="control_derived",
                screen_module="1116",
                score_source="calibrated_beta3",
            ),
        ),
    ]
    for name, query in cases:
        url = "http://127.0.0.1:8554/?" + urllib.parse.urlencode(query)
        browser.call("Page.navigate", dict(url=url))
        deadline = time.monotonic() + 120
        state = {}
        while time.monotonic() < deadline:
            state = (
                browser.js(
                    "JSON.parse(JSON.stringify({ready:document.body.innerText.includes('Independent target candidates'),plots:document.querySelectorAll('.js-plotly-plot').length,errors:document.querySelectorAll('[data-testid=stException]').length,meta:document.querySelector('.js-plotly-plot')?.layout?.meta}))"
                )
                or {}
            )
            meta = state.get("meta") or {}
            if (
                state.get("ready")
                and state.get("plots")
                and meta.get("family") == query["screen_partition"]
                and meta.get("module") == int(query["screen_module"])
                and meta.get("beta")
                == (3 if query["score_source"].endswith("3") else 1)
            ):
                break
            if state.get("errors"):
                raise RuntimeError("Application exception during screenshot capture")
            time.sleep(0.5)
        else:
            raise TimeoutError("Rendered plot did not become ready: " + name)
        time.sleep(2)
        result = browser.call(
            "Page.captureScreenshot",
            dict(
                format="png",
                captureBeyondViewport=False,
                clip=dict(x=0, y=0, width=1920, height=1800, scale=1),
            ),
        )
        path = args.output / (name + ".png")
        path.write_bytes(base64.b64decode(result["data"]))
        proof.append(dict(name=name, url=url, ready=state, path=str(path)))
        print(json.dumps(proof[-1]), flush=True)
    (args.output / "browser_checks.json").write_text(json.dumps(proof, indent=2) + "\n")


if __name__ == "__main__":
    main()
