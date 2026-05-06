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
    /control/adaptive     0|1         -- toggle adaptive params (default 1)

OSC out (client port 9001):
    /gen/noteon   pitch velocity channel
    /gen/noteoff  pitch channel
    /gen/status   string
    /gen/params   temperature top_p   -- current sampling params
"""

import sys
import time
import logging
import threading
import argparse
from pathlib import Path

import json
import torch
from transformers import AutoModelForCausalLM
from pythonosc import dispatcher as osc_dispatcher
from pythonosc import osc_server
from pythonosc.udp_client import SimpleUDPClient

# ── make the anticipation package importable when running from server/ ──────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anticipation import ops
from anticipation.config import TIME_RESOLUTION, MAX_PITCH, MAX_DUR, MAX_TIME
from anticipation.vocab import TIME_OFFSET, DUR_OFFSET, NOTE_OFFSET
from anticipation.sample import generate

from shimon_filter import filter_notes, octave_fold, expand_tremolo, nudge_runs, stagger_chords

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Model loader (handles full checkpoints and LoRA adapters) ─────────────────

def _load_model(model_path: str, device: torch.device):
    adapter_cfg_path = Path(model_path) / "adapter_config.json"
    if adapter_cfg_path.exists():
        from peft import PeftModel
        with open(adapter_cfg_path) as f:
            adapter_cfg = json.load(f)
        base_path = adapter_cfg["base_model_name_or_path"]
        if not Path(base_path).exists():
            # fall back: look for the base model next to the adapter directory
            base_path = str(Path(model_path).parent / "music-medium-800k")
            log.warning("adapter base path not found; trying %s", base_path)
        log.info("LoRA adapter – loading base model from %s …", base_path)
        base = AutoModelForCausalLM.from_pretrained(base_path).to(device)
        log.info("Applying LoRA weights from %s …", model_path)
        model = PeftModel.from_pretrained(base, model_path)
        model = model.merge_and_unload()
        log.info("LoRA weights merged into base model")
        return model
    return AutoModelForCausalLM.from_pretrained(model_path).to(device)


# ── Note buffer ──────────────────────────────────────────────────────────────

class NoteBuffer:
    """Thread-safe buffer that tracks note-on/off events from the human player."""

    def __init__(self):
        self._lock = threading.Lock()
        self._pending: dict[int, tuple[float, int]] = {}   # pitch → (on_time, velocity)
        self._done: list[tuple[float, float, int, int]] = []  # (t, dur, pitch, instrument)
        self._t0: float | None = None
        self._last_event_time: float | None = None          # session-relative time of last note activity

    def start(self):
        with self._lock:
            self._pending.clear()
            self._done.clear()
            self._t0 = time.time()
            self._last_event_time = None

    def note_event(self, pitch: int, velocity: int, instrument: int = 0):
        with self._lock:
            if self._t0 is None:
                return
            now = time.time() - self._t0
            self._last_event_time = now
            if velocity > 0:
                self._pending[pitch] = (now, velocity)
            else:
                if pitch in self._pending:
                    on_t, _ = self._pending.pop(pitch)
                    dur = max(0.05, now - on_t)
                    self._done.append((on_t, dur, pitch, instrument))

    def last_event_time(self) -> float | None:
        """Session-relative time of the most recent note-on or note-off."""
        with self._lock:
            return self._last_event_time

    def collect_window(self, t_start: float, t_end: float) -> list[tuple]:
        """Return completed notes whose note-on fell in [t_start, t_end], window-relative times."""
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

def notes_to_events(notes: list[tuple]) -> list[int]:
    """Convert (t_rel, dur, pitch, instrument) list → regular AMT event tokens.

    Passed as inputs= so the model treats them as already-happened music
    and generates a continuation for the next window.
    """
    events = []
    for (t, dur, pitch, instr) in notes:
        t_bins = min(int(t * TIME_RESOLUTION), MAX_TIME - 1)
        d_bins = min(int(dur * TIME_RESOLUTION), MAX_DUR - 1)
        note_v = pitch + instr * MAX_PITCH
        events.extend([
            TIME_OFFSET + t_bins,
            DUR_OFFSET  + d_bins,
            NOTE_OFFSET + note_v,
        ])
    return ops.sort(events)


def decode_events(events: list[int]) -> list[tuple]:
    """Decode AMT event tokens → sorted list of (t, dur, pitch, instrument)."""
    notes = []
    for i in range(0, len(events), 3):
        t_bins = events[i]     - TIME_OFFSET
        d_bins = events[i + 1] - DUR_OFFSET
        note_v = events[i + 2] - NOTE_OFFSET

        if note_v < 0 or note_v >= MAX_PITCH * 129:
            continue

        instrument = note_v // MAX_PITCH
        pitch      = note_v  % MAX_PITCH
        t_sec      = t_bins / TIME_RESOLUTION
        d_sec      = max(0.05, d_bins / TIME_RESOLUTION)
        notes.append((t_sec, d_sec, pitch, instrument))

    notes.sort(key=lambda x: x[0])
    return notes



def notes_to_schedule(notes: list[tuple], play_start: float, win_start: float = 0.0) -> list[tuple]:
    """Convert decoded (t, dur, pitch, instrument) notes → sorted OSC schedule.

    play_start : wall-clock time to begin playback
    win_start  : session time (seconds) of the window start
    """
    schedule = []
    for (t_sec, d_sec, pitch, instrument) in notes:
        channel  = (instrument % 15) + 2
        on_time  = play_start + (t_sec - win_start)
        off_time = on_time + d_sec
        schedule.append((on_time,  "/gen/noteon",  [pitch, 80, channel]))
        schedule.append((off_time, "/gen/noteoff", [pitch, channel]))

    schedule.sort(key=lambda x: x[0])
    return schedule


def events_to_schedule(events: list[int], play_start: float, win_start: float = 0.0) -> list[tuple]:
    """Convenience wrapper: decode_events → notes_to_schedule (no filtering)."""
    return notes_to_schedule(decode_events(events), play_start, win_start)


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
        phrase_end_silence: float = 1.5,
        max_phrase_duration: float = 5.0,
        silence_timeout: float = 8.0,
        top_p: float = 0.90,
        temperature: float = 0.8,
        adaptive_params: bool = True,
        human_instrument: int = 0,
        min_note_dist_ms: float = 50,
        max_notes_per_onset: int = 4,
        stagger_ms: float = 11.0,
        pitch_lo: int = 48,
        pitch_hi: int = 95,
        max_note_dur_s: float = 1.0,
        tremolo_rate: float = 10.0,
        tremolo_strike_dur_ms: float = 50.0,
        run_interval_ms: float = 150.0,
        run_semitones: int = 3,
        shimonize: bool = True,
    ):
        log.info("Loading model from %s …", model_path)
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.model = _load_model(model_path, device)
        self.model.eval()
        log.info("Model ready on %s", next(self.model.parameters()).device)

        self.client = SimpleUDPClient(client_ip, client_port)
        self.listen_ip   = listen_ip
        self.listen_port = listen_port

        self.buffer                = NoteBuffer()
        self.window_size           = window_size
        self.phrase_end_silence    = phrase_end_silence
        self.max_phrase_duration   = max_phrase_duration
        self.silence_timeout       = silence_timeout
        self.top_p                 = top_p
        self.temperature           = temperature
        self.adaptive_params       = adaptive_params
        self.human_instrument      = human_instrument
        self.min_note_dist_ms      = min_note_dist_ms
        self.max_notes_per_onset   = max_notes_per_onset
        self.stagger_ms            = stagger_ms
        self.pitch_lo              = pitch_lo
        self.pitch_hi              = pitch_hi
        self.max_note_dur_s        = max_note_dur_s
        self.tremolo_rate          = tremolo_rate
        self.tremolo_strike_dur_ms = tremolo_strike_dur_ms
        self.run_interval_ms       = run_interval_ms
        self.run_semitones         = run_semitones
        self.shimonize             = shimonize

        self._running = False
        self._current_temperature = temperature
        self._current_top_p = top_p
        self._prev_pitches = []

    # ── OSC handlers ──────────────────────────────────────────────────────────

    def _on_any(self, address, *args):
        """Catch-all: log every OSC message that arrives."""
        log.info("OSC IN  %s  args=%s", address, args)

    def _on_note(self, address, pitch, velocity):
        label = "ON " if int(velocity) > 0 else "OFF"
        log.info("MIDI %s  pitch=%d  vel=%d", label, int(pitch), int(velocity))
        # auto-start session on first note if not already running
        if not self._running:
            log.info("Auto-starting session on first note")
            self._on_start(address)
        self.buffer.note_event(int(pitch), int(velocity), self.human_instrument)

    def _on_start(self, address, *args):
        if self._running:
            log.info("Already running – ignoring /control/start")
            return
        self._running = True
        self.buffer.start()
        threading.Thread(target=self._generation_loop, daemon=True).start()
        self.client.send_message("/gen/status", ["started"])
        log.info("Session started  window=%.1fs  top_p=%.2f  temp=%.2f",
                 self.window_size, self.top_p, self.temperature)

    def _on_stop(self, address, *args):
        self._running = False
        self.client.send_message("/gen/status", ["session stopped"])
        log.info("Session stopped")

    def _on_test(self, address, *args):
        """Send a C major arpeggio to verify the return path works."""
        log.info("TEST: sending notes to %s:%d …", self.client._address, self.client._port)
        test_notes = [50, 62, 77, 86]  # C4 E4 G4 C5
        def _fire():
            for i, pitch in enumerate(test_notes):
                time.sleep(i * 0.3)
                self.client.send_message("/gen/noteon",  [pitch, 100, 2])
                log.info("TEST → /gen/noteon [pitch=%d vel=100 ch=2]", pitch)
                time.sleep(0.25)
                self.client.send_message("/gen/noteoff", [pitch, 2])
            log.info("TEST done")
        threading.Thread(target=_fire, daemon=True).start()

    def _on_window_size(self, address, value):
        self.window_size = float(value)
        log.info("window_size → %.2f s", self.window_size)

    def _on_top_p(self, address, value):
        self.top_p = float(value)
        log.info("top_p → %.3f", self.top_p)

    def _on_temperature(self, address, value):
        self.temperature = float(value)
        log.info("temperature → %.3f", self.temperature)

    def _on_adaptive(self, address, value):
        self.adaptive_params = bool(int(value))
        log.info("adaptive_params → %s", self.adaptive_params)

    # ── Adaptive sampling ───────────────────────────────────────────────────

    def _repetition_score(self, notes: list[tuple]) -> float:
        """Compare current phrase's pitch sequence to the previous one.

        Returns [0, 1] where 1 = exact repeat. Uses LCS ratio on pitch classes.
        """
        pitches = [p % 12 for (_, _, p, _) in sorted(notes, key=lambda n: n[0])]
        if not self._prev_pitches or not pitches:
            self._prev_pitches = pitches
            return 0.0

        # Longest common subsequence ratio
        m, n = len(self._prev_pitches), len(pitches)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if self._prev_pitches[i - 1] == pitches[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        score = dp[m][n] / max(m, n)
        self._prev_pitches = pitches
        return score

    def _sampling_params(self, notes: list[tuple], phrase_dur: float):
        """Return (temperature, top_p) driven by note density, with a
        repetition threshold override.

        - Sparse playing → higher temp; dense playing → lower temp.
        - Cross-phrase repetition > 0.7 → temperature override to 1.19.
        """
        if not self.adaptive_params or len(notes) < 2:
            return self._current_temperature, self._current_top_p

        rep_score = self._repetition_score(notes)

        density = len(notes) / max(phrase_dur, 0.1)
        density_norm = max(0.0, min(1.0, (density - 0.5) / 5.5))   # sparse=0, dense=1

        temperature = 0.8 + density_norm * 0.4   # [0.8, 1.2]

        if rep_score > 0.7:
            temperature = 1.19

        top_p = 0.90 + (temperature - 0.8) / 0.4 * 0.1   # [0.90, 1.00]

        self._current_temperature = round(temperature, 3)
        self._current_top_p = round(top_p, 3)

        trigger = "repetition" if rep_score > 0.7 else "density"
        log.info("Params: density=%.1f/s rep=%.2f trigger=%s → temp=%.3f top_p=%.3f",
                 density, rep_score, trigger,
                 self._current_temperature, self._current_top_p)
        return self._current_temperature, self._current_top_p

    # ── Generation loop ───────────────────────────────────────────────────────

    def _wait_for_phrase_end(self) -> tuple[float, float]:
        """Block until the user pauses for phrase_end_silence seconds or max_phrase_duration is hit.

        Returns (phrase_start, phrase_end) as session-relative times.
        Also fires /gen/status "rest" once if the user is silent for silence_timeout seconds.
        """
        # wait for the first note that arrives after we start listening
        wait_start = self.buffer.elapsed()
        idle_fired = False
        while self._running:
            last_t = self.buffer.last_event_time()
            if last_t is not None and last_t > wait_start:
                break
            # detect prolonged user silence (Max should stop the gesture)
            if not idle_fired and last_t is not None:
                idle = self.buffer.elapsed() - last_t
                if idle >= self.silence_timeout:
                    log.info("User idle %.1fs ≥ %.1fs → /gen/status rest", idle, self.silence_timeout)
                    self.client.send_message("/gen/status", ["rest"])
                    idle_fired = True
            time.sleep(0.05)

        phrase_start = self.buffer.last_event_time()
        log.info("Phrase started at %.2fs", phrase_start)

        while self._running:
            time.sleep(0.05)
            last_t   = self.buffer.last_event_time()
            elapsed  = self.buffer.elapsed()
            silence  = elapsed - last_t
            duration = last_t - phrase_start

            if silence >= self.phrase_end_silence:
                log.info("Phrase ended — silence %.2fs >= %.2fs", silence, self.phrase_end_silence)
                return phrase_start, last_t

            if duration >= self.max_phrase_duration:
                log.info("Phrase capped — duration %.2fs >= %.2fs", duration, self.max_phrase_duration)
                return phrase_start, last_t

        return phrase_start, self.buffer.elapsed()

    def _generation_loop(self):
        window_num = 0

        while self._running:
            # wait for a complete phrase (first note → silence gap)
            phrase_start, phrase_end = self._wait_for_phrase_end()
            if not self._running:
                break

            phrase_dur = phrase_end - phrase_start
            log.info("Window %d: collecting phrase [%.1f, %.1f]s (%.1fs)",
                     window_num, phrase_start, phrase_end, phrase_dur)
            notes = self.buffer.collect_window(phrase_start, phrase_end)

            if not notes:
                log.info("Window %d: no notes in phrase – skipping", window_num)
                window_num += 1
                continue

            log.info("Window %d: %d notes – generating …", window_num, len(notes))
            t_gen_start = time.time()

            try:
                prompt = notes_to_events(notes)

                note_names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
                log.info("── PROMPT ────────────────────────────────────────")
                for (t, dur, pitch, instr) in sorted(notes, key=lambda x: x[0]):
                    name = note_names[pitch % 12] + str(pitch // 12 - 1)
                    log.info("  t=%5.2fs  dur=%.2fs  %s (pitch=%d  instr=%d)",
                             t, dur, name, pitch, instr)
                log.info("  %d notes → continuation [0, %.1f]s", len(notes), phrase_dur)
                log.info("─────────────────────────────────────────────────")

                temperature, top_p = self._sampling_params(notes, phrase_dur)
                self.client.send_message("/gen/params", [temperature, top_p])
                # only retrigger the regular gesture when below the special-gesture threshold
                if temperature < 1.2:
                    self.client.send_message("/gen/status", ["played"])
                with torch.no_grad():
                    events = generate(
                        self.model,
                        start_time  = phrase_dur,
                        end_time    = phrase_dur * 2,
                        inputs      = prompt,
                        controls    = [],
                        top_p       = top_p,
                        temperature = temperature,
                    )
            except Exception as exc:
                log.exception("Generation failed: %s", exc)
                self.client.send_message("/gen/status", [f"error: {exc}"])
                window_num += 1
                continue

            # clip to just the new continuation window
            continuation = ops.clip(events, phrase_dur, phrase_dur * 2,
                                    clip_duration=False, seconds=True)

            gen_elapsed = time.time() - t_gen_start
            n_events = len(continuation) // 3
            log.info("Window %d: generated %d events in %.2fs", window_num, n_events, gen_elapsed)

            # decode → (optional shimonization) → schedule
            play_start = time.time()
            t0 = time.time()

            decoded = decode_events(continuation)
            t1 = time.time(); log.info("  pipeline  decode_events  : %5.3f ms  (%d notes)", (t1-t0)*1e3, len(decoded))

            if self.shimonize:
                decoded = octave_fold(decoded, self.pitch_lo, self.pitch_hi)
                t2 = time.time(); log.info("  pipeline  octave_fold    : %5.3f ms", (t2-t1)*1e3)

                decoded = expand_tremolo(decoded, self.max_note_dur_s,
                                         self.tremolo_rate, self.tremolo_strike_dur_ms)
                t3 = time.time(); log.info("  pipeline  expand_tremolo : %5.3f ms  (%d notes)", (t3-t2)*1e3, len(decoded))

                decoded = stagger_chords(decoded, self.stagger_ms)
                t4 = time.time(); log.info("  pipeline  stagger_chords : %5.3f ms", (t4-t3)*1e3)

                decoded = nudge_runs(decoded, self.run_interval_ms, self.run_semitones)
                t5 = time.time(); log.info("  pipeline  nudge_runs     : %5.3f ms", (t5-t4)*1e3)

                decoded = filter_notes(decoded, self.min_note_dist_ms, self.max_notes_per_onset)
                t6 = time.time(); log.info("  pipeline  filter_notes   : %5.3f ms  (%d notes)", (t6-t5)*1e3, len(decoded))

                log.info("  pipeline  TOTAL (shim)   : %5.3f ms", (t6-t0)*1e3)
            else:
                log.info("  pipeline  shimonize=False – skipping transforms")

            schedule = notes_to_schedule(decoded, play_start, phrase_dur)
            t7 = time.time(); log.info("  pipeline  notes_to_sched : %5.3f ms", (t7-t1)*1e3)
            threading.Thread(
                target=self._playback_thread,
                args=(schedule,),
                daemon=True,
            ).start()

            window_num += 1

    def _playback_thread(self, schedule: list[tuple]):
        log.info("Playback: sending %d OSC messages to %s:%d",
                 len(schedule), self.client._address, self.client._port)
        for (target_time, address, args) in schedule:
            now = time.time()
            if target_time > now:
                time.sleep(target_time - now)
            log.info("  → %s %s", address, args)
            self.client.send_message(address, args)
        log.info("Playback: done")

    # ── Start OSC server ──────────────────────────────────────────────────────

    def run(self):
        disp = osc_dispatcher.Dispatcher()
        disp.map("/note",                   self._on_note)
        disp.map("/control/start",          self._on_start)
        disp.map("/control/stop",           self._on_stop)
        disp.map("/control/window_size",    self._on_window_size)
        disp.map("/control/top_p",          self._on_top_p)
        disp.map("/control/temperature",    self._on_temperature)
        disp.map("/control/adaptive",       self._on_adaptive)
        disp.map("/control/test",           self._on_test)
        disp.set_default_handler(self._on_any)

        server = osc_server.ThreadingOSCUDPServer(
            (self.listen_ip, self.listen_port), disp
        )
        log.info("OSC server listening on %s:%d", self.listen_ip, self.listen_port)
        log.info("Sending generated notes to %s:%d",
                 self.client._address, self.client._port)
        threading.Thread(target=self._startup_test, daemon=True).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            log.info("Shutting down – sending all notes off …")
            self._running = False
            for ch in range(2, 17):
                for pitch in range(128):
                    self.client.send_message("/gen/noteoff", [pitch, ch])
            self.client.send_message("/gen/status", ["stopped"])
            log.info("Done")

    def _startup_test(self):
        self._on_start("/gen/status/started")
        time.sleep(2.0)
        log.info("STARTUP TEST: firing C major arpeggio to %s:%d",
                 self.client._address, self.client._port)
        for pitch in [50, 62, 77, 86]:
            self.client.send_message("/gen/noteon",  [pitch, 100, 2])
            log.info("  → /gen/noteon [pitch=%d vel=100 ch=2]", pitch)
            time.sleep(0.3)
            self.client.send_message("/gen/noteoff", [pitch, 2])
        log.info("STARTUP TEST done – if Max heard 4 notes the return path is working")
        


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    # parser = argparse.ArgumentParser(description="AMT real-time jam server")
    # parser.add_argument("--listen-ip",   default="0.0.0.0",
    #                     help="IP to bind OSC server (default 0.0.0.0)")
    # parser.add_argument("--listen-port", type=int, default=9000,
    #                     help="UDP port to listen on (default 9000)")
    # parser.add_argument("--client-ip",   required=True,
    #                     help="Public IP of the local machine to send generated notes to")
    # parser.add_argument("--client-port", type=int, default=9001,
    #                     help="UDP port on local machine (default 9001)")
    # parser.add_argument("--model",       default="../model/music-small-800k",
    #                     help="Path to model checkpoint (default model/music-small-800k)")
    # parser.add_argument("--window",      type=float, default=6.0,
    #                     help="Window size in seconds (default 6.0)")
    # parser.add_argument("--top-p",       type=float, default=0.95,
    #                     help="Nucleus sampling p (default 0.95)")
    # parser.add_argument("--temperature", type=float, default=1.0,
    #                     help="Sampling temperature (default 1.0
    # args = parser.parse_args()

    # model_path = '/data/AMTmodel/music-medium-800k'
    model_path = '/data/AMTmodel/full-18000'
    client_ip = "192.168.1.2"
    listen_ip = "192.168.1.10"
    client_port = 9001
    listen_port = 9000

    server = JamServer(
        model_path    = model_path,
        listen_ip     = listen_ip,
        listen_port   = listen_port,
        client_ip     = client_ip,
        client_port   = client_port,
        window_size   = 4.0,
        top_p         = 0.90,
        temperature   = 0.8,
        shimonize= True
    )
    server.run()


if __name__ == "__main__":
    main()
