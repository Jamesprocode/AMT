"""
AMT Jam Server
Runs on the GPU machine. Receives human MIDI notes via OSC,
generates accompaniment with the Anticipatory Music Transformer,
and sends the generated notes back as OSC.

Usage:
    python jam_server.py --client-ip <local_machine_ip>

OSC in  (port 9000):
    /note         pitch velocity      -- velocity=0 means note-off
    /control/start
    /control/stop
    /control/window_size  float       -- seconds per window (default 6.0)
    /control/top_p        float       -- nucleus sampling (default 0.95)
    /control/temperature  float       -- sampling temperature (default 1.0)

OSC out (client port 9001):
    /gen/noteon   pitch velocity channel
    /gen/noteoff  pitch channel
    /gen/status   string
"""

import sys
import time
import logging
import threading
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM
from pythonosc import dispatcher as osc_dispatcher
from pythonosc import osc_server
from pythonosc.udp_client import SimpleUDPClient

# ── make the anticipation package importable when running from server/ ──────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anticipation import ops
from anticipation.config import TIME_RESOLUTION, MAX_PITCH, MAX_DUR, MAX_TIME, DELTA
from anticipation.vocab import (
    TIME_OFFSET, DUR_OFFSET, NOTE_OFFSET,
    ATIME_OFFSET, ADUR_OFFSET, ANOTE_OFFSET,
    CONTROL_OFFSET,
)
from anticipation.sample import generate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Note buffer ──────────────────────────────────────────────────────────────

class NoteBuffer:
    """Thread-safe buffer that tracks note-on/off events from the human player."""

    def __init__(self):
        self._lock = threading.Lock()
        self._pending: dict[int, tuple[float, int]] = {}   # pitch → (on_time, velocity)
        self._done: list[tuple[float, float, int, int]] = []  # (t, dur, pitch, instrument)
        self._t0: float | None = None

    def start(self):
        with self._lock:
            self._pending.clear()
            self._done.clear()
            self._t0 = time.time()

    def note_event(self, pitch: int, velocity: int, instrument: int = 0):
        with self._lock:
            if self._t0 is None:
                return
            now = time.time() - self._t0
            if velocity > 0:
                self._pending[pitch] = (now, velocity)
            else:
                if pitch in self._pending:
                    on_t, _ = self._pending.pop(pitch)
                    dur = max(0.05, now - on_t)
                    self._done.append((on_t, dur, pitch, instrument))

    def collect_window(self, t_start: float, t_end: float) -> list[tuple]:
        """Return completed notes whose note-on fell in [t_start, t_end].
        Also closes any still-pending notes using t_end as note-off time."""
        with self._lock:
            notes = []
            for (t, dur, pitch, instr) in self._done:
                if t_start <= t < t_end:
                    notes.append((t - t_start, dur, pitch, instr))

            # close any notes still held at window boundary
            for pitch, (on_t, _) in list(self._pending.items()):
                if t_start <= on_t < t_end:
                    dur = max(0.05, t_end - on_t)
                    notes.append((on_t - t_start, dur, pitch, 0))

            return notes

    def elapsed(self) -> float:
        if self._t0 is None:
            return 0.0
        return time.time() - self._t0


# ── Token helpers ─────────────────────────────────────────────────────────────

def notes_to_controls(notes: list[tuple]) -> list[int]:
    """Convert (t_rel, dur, pitch, instrument) list → AMT control tokens.

    Times are relative to the window start (0..window_size).
    We shift them by DELTA so they appear as 'future' events the model
    will anticipate while generating [0, window_size].
    """
    controls = []
    for (t, dur, pitch, instr) in notes:
        t_shifted = t + DELTA                              # push into future
        t_bins  = min(int(t_shifted * TIME_RESOLUTION), MAX_TIME - 1)
        d_bins  = min(int(dur       * TIME_RESOLUTION), MAX_DUR  - 1)
        note_v  = pitch + instr * MAX_PITCH
        controls.extend([
            ATIME_OFFSET + t_bins,
            ADUR_OFFSET  + d_bins,
            ANOTE_OFFSET + note_v,
        ])
    return ops.sort(controls)


def events_to_schedule(events: list[int], play_start: float) -> list[tuple]:
    """Convert AMT event tokens → sorted list of (wall_time, address, args).

    Returns both noteon and noteoff entries so Max/MSP just plays them
    as they arrive with no additional scheduling needed.
    """
    schedule = []
    for i in range(0, len(events), 3):
        t_bins = events[i]     - TIME_OFFSET
        d_bins = events[i + 1] - DUR_OFFSET
        note_v = events[i + 2] - NOTE_OFFSET

        if note_v < 0 or note_v >= MAX_PITCH * 129:
            continue

        instrument = note_v // MAX_PITCH
        pitch      = note_v  % MAX_PITCH
        t_sec  = t_bins / TIME_RESOLUTION
        d_sec  = max(0.05, d_bins / TIME_RESOLUTION)

        # Map instrument → MIDI channel 2-16 (channel 1 reserved for human)
        channel = (instrument % 15) + 2

        on_time  = play_start + t_sec
        off_time = on_time + d_sec

        schedule.append((on_time,  "/gen/noteon",  [pitch, 80, channel]))
        schedule.append((off_time, "/gen/noteoff", [pitch, channel]))

    schedule.sort(key=lambda x: x[0])
    return schedule


# ── Jam server ────────────────────────────────────────────────────────────────

class JamServer:
    def __init__(
        self,
        model_path: str,
        listen_ip: str,
        listen_port: int,
        client_ip: str,
        client_port: int,
        window_size: float = 6.0,
        top_p: float = 0.95,
        temperature: float = 1.0,
        human_instrument: int = 0,
    ):
        log.info("Loading model from %s …", model_path)
        device = torch.device("cuda:1")
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        self.model.eval()
        log.info("Model ready on %s", next(self.model.parameters()).device)

        self.client = SimpleUDPClient(client_ip, client_port)
        self.listen_ip   = listen_ip
        self.listen_port = listen_port

        self.buffer          = NoteBuffer()
        self.window_size     = window_size
        self.top_p           = top_p
        self.temperature     = temperature
        self.human_instrument = human_instrument

        self._running = False

    # ── OSC handlers ──────────────────────────────────────────────────────────

    def _on_note(self, address, pitch, velocity):
        self.buffer.note_event(int(pitch), int(velocity), self.human_instrument)

    def _on_start(self, address, *args):
        if self._running:
            log.info("Already running – ignoring /control/start")
            return
        self._running = True
        self.buffer.start()
        threading.Thread(target=self._generation_loop, daemon=True).start()
        self.client.send_message("/gen/status", ["session started"])
        log.info("Session started  window=%.1fs  top_p=%.2f  temp=%.2f",
                 self.window_size, self.top_p, self.temperature)

    def _on_stop(self, address, *args):
        self._running = False
        self.client.send_message("/gen/status", ["session stopped"])
        log.info("Session stopped")

    def _on_window_size(self, address, value):
        self.window_size = float(value)
        log.info("window_size → %.2f s", self.window_size)

    def _on_top_p(self, address, value):
        self.top_p = float(value)
        log.info("top_p → %.3f", self.top_p)

    def _on_temperature(self, address, value):
        self.temperature = float(value)
        log.info("temperature → %.3f", self.temperature)

    # ── Generation loop ───────────────────────────────────────────────────────

    def _generation_loop(self):
        window_num = 0
        while self._running:
            # wait for one full window of human input
            time.sleep(self.window_size)
            if not self._running:
                break

            elapsed   = self.buffer.elapsed()
            win_end   = elapsed
            win_start = win_end - self.window_size

            log.info("Window %d: collecting [%.1f, %.1f]s", window_num, win_start, win_end)
            notes = self.buffer.collect_window(win_start, win_end)

            if not notes:
                log.info("Window %d: no human notes – skipping generation", window_num)
                window_num += 1
                continue

            log.info("Window %d: %d notes captured – generating …", window_num, len(notes))
            t_gen_start = time.time()

            try:
                controls  = notes_to_controls(notes)
                with torch.no_grad():
                    events = generate(
                        self.model,
                        start_time  = 0,
                        end_time    = self.window_size,
                        inputs      = [],
                        controls    = controls,
                        top_p       = self.top_p,
                        temperature = self.temperature,
                    )
            except Exception as exc:
                log.exception("Generation failed: %s", exc)
                self.client.send_message("/gen/status", [f"error: {exc}"])
                window_num += 1
                continue

            gen_elapsed = time.time() - t_gen_start
            n_events = len(events) // 3
            log.info("Window %d: generated %d events in %.2fs", window_num, n_events, gen_elapsed)

            # play back starting NOW (generation time is the only latency)
            play_start = time.time()
            schedule   = events_to_schedule(events, play_start)
            threading.Thread(
                target=self._playback_thread,
                args=(schedule,),
                daemon=True,
            ).start()

            window_num += 1

    def _playback_thread(self, schedule: list[tuple]):
        for (target_time, address, args) in schedule:
            now = time.time()
            if target_time > now:
                time.sleep(target_time - now)
            self.client.send_message(address, args)

    # ── Start OSC server ──────────────────────────────────────────────────────

    def run(self):
        disp = osc_dispatcher.Dispatcher()
        disp.map("/note",                   self._on_note)
        disp.map("/control/start",          self._on_start)
        disp.map("/control/stop",           self._on_stop)
        disp.map("/control/window_size",    self._on_window_size)
        disp.map("/control/top_p",          self._on_top_p)
        disp.map("/control/temperature",    self._on_temperature)

        server = osc_server.ThreadingOSCUDPServer(
            (self.listen_ip, self.listen_port), disp
        )
        log.info("OSC server listening on %s:%d", self.listen_ip, self.listen_port)
        log.info("Sending generated notes to %s:%d",
                 self.client._address, self.client._port)
        server.serve_forever()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="AMT real-time jam server")
    parser.add_argument("--listen-ip",   default="0.0.0.0",
                        help="IP to bind OSC server (default 0.0.0.0)")
    parser.add_argument("--listen-port", type=int, default=9000,
                        help="UDP port to listen on (default 9000)")
    parser.add_argument("--client-ip",   required=True,
                        help="Public IP of the local machine to send generated notes to")
    parser.add_argument("--client-port", type=int, default=9001,
                        help="UDP port on local machine (default 9001)")
    parser.add_argument("--model",       default="../model/music-small-800k",
                        help="Path to model checkpoint (default model/music-small-800k)")
    parser.add_argument("--window",      type=float, default=6.0,
                        help="Window size in seconds (default 6.0)")
    parser.add_argument("--top-p",       type=float, default=0.95,
                        help="Nucleus sampling p (default 0.95)")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (default 1.0)")
    args = parser.parse_args()

    model_path = 'model/music-small-800k'
    client_ip = "143.215.16.231"
    client_port = 9001

    server = JamServer(
        model_path    = model_path,
        listen_ip     = args.listen_ip,
        listen_port   = args.listen_port,
        client_ip     = client_ip,
        client_port   = client_port,
        window_size   = 6.0,
        top_p         = 0.95,
        temperature   = 1.0,
    )
    server.run()


if __name__ == "__main__":
    main()
