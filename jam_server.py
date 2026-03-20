"""
AMT Jam Server — Bar-Level Comping with Beat Detection
Runs on the GPU machine. Receives human MIDI notes via OSC,
detects tempo via librosa/madmom, generates accompaniment with
the Anticipatory Music Transformer using a pipeline architecture
(controls → events flow across generation cycles), and sends
generated notes back as OSC.

Architecture:
    - Beat detection estimates BPM from note onsets (librosa plp + madmom)
    - Bar-level windowing: 14 bars history + 2 bars controls = 16 bars context
    - Pipeline: user melody flows controls → events across generations
    - Always ANTICIPATE mode (user melody as controls, model generates accomp)
    - Trigger: generate next 2 bars when ~1 bar of accompaniment remains

Usage:
    python jam_server.py --client-ip <local_machine_ip>

OSC in  (port 9000):
    /note         pitch velocity      -- velocity=0 means note-off
    /control/start
    /control/stop
    /control/default_bpm  float       -- manual BPM override (default 120.0)
    /control/top_p        float       -- nucleus sampling (default 0.95)
    /control/temperature  float       -- sampling temperature (default 1.0)

OSC out (client port 9001):
    /gen/noteon   pitch velocity channel
    /gen/noteoff  pitch channel
    /gen/status   string
    /gen/role     role_label improv_score gen_mode
"""

import sys
import time
import logging
import threading
from pathlib import Path

import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from pythonosc import dispatcher as osc_dispatcher
from pythonosc import osc_server
from pythonosc.udp_client import SimpleUDPClient

import librosa

# ── make the anticipation package importable when running from server/ ──────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anticipation import ops
from anticipation.config import TIME_RESOLUTION, MAX_PITCH, MAX_DUR, MAX_TIME, DELTA
from anticipation.vocab import TIME_OFFSET, DUR_OFFSET, NOTE_OFFSET, ATIME_OFFSET, ADUR_OFFSET, ANOTE_OFFSET
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


# ── Beat tracker ─────────────────────────────────────────────────────────────

class BeatTracker:
    """Estimate BPM from note onset times using librosa plp() and madmom."""

    def __init__(self, default_bpm: float = 120.0, min_bpm: float = 50.0,
                 max_bpm: float = 240.0, beats_per_bar: int = 4,
                 fps: int = 100, ema_alpha: float = 0.3):
        self._lock = threading.Lock()
        self._onsets: list[float] = []
        self._bpm = default_bpm
        self.default_bpm = default_bpm
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm
        self.beats_per_bar = beats_per_bar
        self.fps = fps
        self.ema_alpha = ema_alpha
        self._ready = False

    def add_onset(self, t: float):
        with self._lock:
            self._onsets.append(t)

    def reset(self):
        with self._lock:
            self._onsets.clear()
            self._bpm = self.default_bpm
            self._ready = False

    @property
    def bpm(self) -> float:
        with self._lock:
            return self._bpm

    @property
    def ready(self) -> bool:
        with self._lock:
            return self._ready

    @property
    def bar_duration(self) -> float:
        return self.beats_per_bar * 60.0 / self.bpm

    def _make_onset_envelope(self, onsets: list[float]) -> np.ndarray:
        """Synthesize onset envelope from onset times at self.fps."""
        if not onsets:
            return np.array([])
        # Make relative to first onset so envelope starts near t=0
        t_min = min(onsets)
        span = max(onsets) - t_min + 1.0
        n_frames = int(span * self.fps) + 1
        env = np.zeros(n_frames, dtype=np.float32)
        for t in onsets:
            frame = int((t - t_min) * self.fps)
            if 0 <= frame < n_frames:
                env[frame] = 1.0
        return env

    def update(self) -> float:
        """Recompute BPM from recent onsets using librosa plp().

        Returns current BPM estimate.
        """
        with self._lock:
            onsets = list(self._onsets[-50:])  # recent onsets only for fast tempo est

        if len(onsets) < 6:
            log.info("BeatTracker: only %d onsets (need 6)", len(onsets))
            return self._bpm

        env = self._make_onset_envelope(onsets)
        if len(env) < self.fps * 2:
            log.info("BeatTracker: envelope too short (%d frames, need %d)", len(env), self.fps * 2)
            return self._bpm

        try:
            # librosa tempo estimation from onset envelope
            # hop_length=1 because each frame in env is one time step at self.fps
            tempo = librosa.feature.tempo(
                onset_envelope=env,
                sr=self.fps,
                hop_length=1,
                start_bpm=self._bpm,
            )
            raw_bpm = float(tempo[0]) if len(tempo) > 0 else self._bpm
            log.info("BeatTracker: raw=%.1f BPM from %d onsets, env=%d frames", raw_bpm, len(onsets), len(env))
        except Exception as exc:
            log.warning("BeatTracker: librosa failed: %s", exc)
            return self._bpm

        # Guard against inf/nan from librosa
        if not np.isfinite(raw_bpm) or raw_bpm <= 0:
            log.warning("BeatTracker: librosa returned invalid BPM (%.1f), skipping", raw_bpm)
            return self._bpm

        # Fold into valid range
        bpm = raw_bpm
        while bpm < self.min_bpm:
            bpm *= 2.0
        while bpm > self.max_bpm:
            bpm /= 2.0

        # EMA smooth
        with self._lock:
            if self._ready:
                self._bpm = self.ema_alpha * bpm + (1 - self.ema_alpha) * self._bpm
            else:
                self._bpm = bpm
                self._ready = True
            return self._bpm


# ── Generated history ────────────────────────────────────────────────────────

class GeneratedHistory:
    """Stores model-generated accompaniment notes in session time."""

    def __init__(self):
        self._lock = threading.Lock()
        self._notes: list[tuple[float, float, int, int]] = []

    def add_notes(self, notes: list[tuple], session_offset: float):
        """Add decoded notes, converting from model-relative to session time."""
        with self._lock:
            for (t, dur, pitch, instr) in notes:
                self._notes.append((session_offset + t, dur, pitch, instr))

    def collect_window(self, t_start: float, t_end: float) -> list[tuple]:
        """Return notes in [t_start, t_end) with window-relative times."""
        with self._lock:
            notes = []
            for (t, dur, pitch, instr) in self._notes:
                if t_start <= t < t_end:
                    notes.append((t - t_start, dur, pitch, instr))
            return notes

    def clear(self):
        with self._lock:
            self._notes.clear()


# ── Note splitting ────────────────────────────────────────────────────────────

def extract_top_melody(notes: list[tuple], onset_window: float = 0.03,
                        melody_instr: int = 52):
    """Extract top note per onset group as melody.

    Groups simultaneous notes (within onset_window seconds), returns the
    highest pitch from each group as melody_instr.

    Returns list of (t, dur, pitch, melody_instr).
    """
    if not notes:
        return []

    sorted_notes = sorted(notes, key=lambda n: n[0])

    # Group by onset time
    groups = []
    current_group = [sorted_notes[0]]
    for note in sorted_notes[1:]:
        if note[0] - current_group[0][0] < onset_window:
            current_group.append(note)
        else:
            groups.append(current_group)
            current_group = [note]
    groups.append(current_group)

    melody = []
    for group in groups:
        top = max(group, key=lambda n: n[2])
        melody.append((top[0], top[1], top[2], melody_instr))

    return melody


# ── Token helpers ─────────────────────────────────────────────────────────────

def notes_to_events(notes: list[tuple]) -> list[int]:
    """Convert (t_rel, dur, pitch, instrument) list → regular AMT event tokens."""
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


def notes_to_controls(notes: list[tuple]) -> list[int]:
    """Convert (t_rel, dur, pitch, instrument) list → anticipatory control tokens.

    Each note is shifted forward by DELTA seconds so the model sees them as
    future constraints and generates accompaniment for the same window.
    """
    controls = []
    for (t, dur, pitch, instr) in notes:
        t_shifted = t + DELTA
        t_bins = min(int(t_shifted * TIME_RESOLUTION), MAX_TIME - 1)
        d_bins = min(int(dur * TIME_RESOLUTION), MAX_DUR - 1)
        note_v = pitch + instr * MAX_PITCH
        controls.extend([
            ATIME_OFFSET + t_bins,
            ADUR_OFFSET  + d_bins,
            ANOTE_OFFSET + note_v,
        ])
    return ops.sort(controls)


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
    win_start  : model time (seconds) corresponding to the start of playback
    """
    schedule = []
    for (t_sec, d_sec, pitch, instrument) in notes:
        # POP909: piano=0→ch0, guitar(24)→ch1, melody(52)→ch2
        channel  = {0: 0, 24: 1, 52: 2}.get(instrument, instrument % 16)
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
        gen_bars: int = 2,
        history_bars: int = 14,
        beats_per_bar: int = 4,
        default_bpm: float = 120.0,
        latency_bars: int = 1,
        key_change_threshold: float = 0.35,
        top_p: float = 0.95,
        temperature: float = 1.0,
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
        self.beat_tracker          = BeatTracker(default_bpm=default_bpm,
                                                  beats_per_bar=beats_per_bar,
                                                  ema_alpha=0.15)
        self.gen_history           = GeneratedHistory()
        self.gen_bars              = gen_bars
        self.history_bars          = history_bars
        self.latency_bars          = latency_bars
        self.key_change_threshold  = key_change_threshold
        self.top_p                 = top_p
        self.temperature           = temperature
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

        self._running                  = False
        self._priming                  = True   # Phase 1: user plays chords as context
        self._prev_pitch_classes: set  = set()
        self._current_playback_cancel: threading.Event | None = None
        self._gen_end_session          = 0.0
        self._last_gen_time            = 0.0
        self.priming_bars              = 4      # bars of user accompaniment before generation

        # ── Role detection state (kept for logging, future switch) ────────
        self._current_role       = "improv"
        self._role_score_ema     = 0.5
        self._role_ema_alpha     = 0.4
        self._role_switch_high   = 0.65
        self._role_switch_low    = 0.35
        self._generation_mode    = "anticipate"
        self._phrases_in_mode    = 0
        self._min_phrases_switch = 2

    # ── OSC handlers ──────────────────────────────────────────────────────────

    def _on_any(self, address, *args):
        """Catch-all: log every OSC message that arrives."""
        log.info("OSC IN  %s  args=%s", address, args)

    def _on_note(self, address, pitch, velocity):
        label = "ON " if int(velocity) > 0 else "OFF"
        log.info("MIDI %s  pitch=%d  vel=%d", label, int(pitch), int(velocity))
        # auto-start session on first note-on (not note-off)
        if not self._running and int(velocity) > 0:
            log.info("Auto-starting session on first note-on")
            self._on_start(address)
        # Feed beat tracker on note-on
        if int(velocity) > 0:
            self.beat_tracker.add_onset(self.buffer.elapsed())
        self.buffer.note_event(int(pitch), int(velocity), self.human_instrument)

    def _on_start(self, address, *args):
        if self._running:
            log.info("Already running – ignoring /control/start")
            return
        self._running = True
        self.buffer.start()
        self.beat_tracker.reset()
        self.gen_history.clear()
        self._gen_end_session = 0.0
        self._last_gen_time = 0.0
        threading.Thread(target=self._generation_loop, daemon=True).start()
        self.client.send_message("/gen/status", ["session started"])
        log.info("Session started  gen=%d bars  history=%d bars  BPM=%.0f  top_p=%.2f  temp=%.2f",
                 self.gen_bars, self.history_bars, self.beat_tracker.bpm,
                 self.top_p, self.temperature)

    def _on_stop(self, address, *args):
        self._running = False
        self.client.send_message("/gen/status", ["session stopped"])
        log.info("Session stopped")

    def _on_test(self, address, *args):
        """Send a C major arpeggio to verify the return path works."""
        log.info("TEST: sending notes to %s:%d …", self.client._address, self.client._port)
        test_notes = [60, 64, 67, 72]  # C4 E4 G4 C5
        def _fire():
            for i, pitch in enumerate(test_notes):
                time.sleep(i * 0.3)
                self.client.send_message("/gen/noteon",  [pitch, 100, 2])
                log.info("TEST → /gen/noteon [pitch=%d vel=100 ch=2]", pitch)
                time.sleep(0.25)
                self.client.send_message("/gen/noteoff", [pitch, 2])
            log.info("TEST done")
        threading.Thread(target=_fire, daemon=True).start()

    def _on_default_bpm(self, address, value):
        self.beat_tracker.default_bpm = float(value)
        log.info("default_bpm → %.1f", float(value))

    def _on_top_p(self, address, value):
        self.top_p = float(value)
        log.info("top_p → %.3f", self.top_p)

    def _on_temperature(self, address, value):
        self.temperature = float(value)
        log.info("temperature → %.3f", self.temperature)

    def _on_prime_done(self, address, *args):
        """User signals they're done playing the initial accompaniment context."""
        self._priming = False
        log.info("Priming ended — switching to comping mode (user melody → controls)")

    # ── Key change detection ──────────────────────────────────────────────────

    def _detect_key_change(self, notes: list[tuple]) -> bool:
        """Return True if the pitch class content has shifted dramatically."""
        current = set(pitch % 12 for (_, _, pitch, _) in notes)
        if not self._prev_pitch_classes or not current:
            self._prev_pitch_classes = current
            return False
        union        = self._prev_pitch_classes | current
        intersection = self._prev_pitch_classes & current
        similarity   = len(intersection) / len(union)
        changed      = similarity < self.key_change_threshold
        if changed:
            log.info("Key change detected — Jaccard=%.2f (threshold=%.2f)", similarity, self.key_change_threshold)
        self._prev_pitch_classes = current
        return changed

    # ── Role detection (kept for logging/OSC feedback) ────────────────────────

    def _repetition_score(self, notes: list[tuple]) -> float:
        """Detect repetition within the current window."""
        pitches = [p % 12 for (_, _, p, _) in sorted(notes, key=lambda n: n[0])]
        n = len(pitches)
        if n < 4:
            return 0.0

        best_score = 0.0
        for motif_len in range(2, n // 2 + 1):
            tail = n - motif_len
            if tail == 0:
                continue
            matches = sum(1 for i in range(motif_len, n) if pitches[i] == pitches[i % motif_len])
            score = matches / tail
            if score > best_score:
                best_score = score

        return best_score

    def _detect_role(self, notes: list[tuple]) -> tuple[float, str]:
        """Classify user input as improvisation vs accompaniment.

        Returns (improv_score, role_label). Currently used for logging only.
        """
        if len(notes) < 3:
            return (0.5, self._current_role)

        sorted_notes = sorted(notes, key=lambda x: x[0])

        # Signal 1: Polyphony
        onset_groups = []
        current_group = [sorted_notes[0]]
        for note in sorted_notes[1:]:
            if note[0] - current_group[-1][0] < 0.03:
                current_group.append(note)
            else:
                onset_groups.append(current_group)
                current_group = [note]
        onset_groups.append(current_group)

        max_group = max(len(g) for g in onset_groups)
        if max_group == 1:
            return (1.0, "improv")

        avg_sim = sum(len(g) for g in onset_groups) / len(onset_groups)
        polyphony_norm = max(0.0, min(1.0, (avg_sim - 1.0) / 2.5))
        improv_from_polyphony = 1.0 - polyphony_norm

        # Signal 2: Repetition
        rep_score = self._repetition_score(notes)
        improv_from_repetition = 1.0 - rep_score

        # Signal 3: Rhythmic regularity
        group_onsets = [g[0][0] for g in onset_groups]
        if len(group_onsets) >= 3:
            iois = [group_onsets[i + 1] - group_onsets[i] for i in range(len(group_onsets) - 1)]
            mean_ioi = sum(iois) / len(iois)
            if mean_ioi > 0:
                std_ioi = (sum((x - mean_ioi) ** 2 for x in iois) / len(iois)) ** 0.5
                cv = std_ioi / mean_ioi
                improv_from_rhythm = max(0.0, min(1.0, cv / 0.8))
            else:
                improv_from_rhythm = 0.5
        else:
            improv_from_rhythm = 0.5

        # Signal 4: Pitch range
        pitches = [pitch for (_, _, pitch, _) in notes]
        pitch_range = max(pitches) - min(pitches)
        range_norm = max(0.0, min(1.0, (pitch_range - 12) / 24.0))
        improv_from_range = 1.0 - range_norm

        # Signal 5: Stepwise motion
        sequential_pitches = [g[0][2] for g in onset_groups]
        if len(sequential_pitches) >= 2:
            intervals = [abs(sequential_pitches[i + 1] - sequential_pitches[i])
                         for i in range(len(sequential_pitches) - 1)]
            stepwise_count = sum(1 for iv in intervals if iv <= 2)
            improv_from_intervals = stepwise_count / len(intervals)
        else:
            improv_from_intervals = 0.5

        improv_score = (0.30 * improv_from_polyphony +
                        0.20 * improv_from_repetition +
                        0.20 * improv_from_rhythm +
                        0.15 * improv_from_range +
                        0.15 * improv_from_intervals)

        role_label = "improv" if improv_score > 0.5 else "accomp"

        log.info("Role: score=%.2f [poly=%.2f rep=%.2f rhy=%.2f rng=%.2f stp=%.2f] → %s",
                 improv_score, improv_from_polyphony, improv_from_repetition,
                 improv_from_rhythm, improv_from_range, improv_from_intervals, role_label)

        return (improv_score, role_label)

    # ── _update_role (kept for future use, not called in comping-only mode) ──

    def _update_role(self, improv_score: float) -> str:
        """EMA smoothing + hysteresis → return generation mode string."""
        self._role_score_ema = (self._role_ema_alpha * improv_score +
                                (1 - self._role_ema_alpha) * self._role_score_ema)

        self._phrases_in_mode += 1

        if self._phrases_in_mode < self._min_phrases_switch:
            return self._generation_mode

        if self._current_role == "accomp" and self._role_score_ema > self._role_switch_high:
            self._current_role = "improv"
            self._generation_mode = "anticipate"
            self._phrases_in_mode = 0
            log.info("Mode switch → ANTICIPATE (user improvising, model accompanies)")
        elif self._current_role == "improv" and self._role_score_ema < self._role_switch_low:
            self._current_role = "accomp"
            self._generation_mode = "autoregress"
            self._phrases_in_mode = 0
            log.info("Mode switch → AUTOREGRESS (user accompanying, model solos)")

        return self._generation_mode

    # ── Generation loop (pipeline architecture) ──────────────────────────────

    def _generation_loop(self):
        window_num = 0
        shift_bars = self.gen_bars + self.latency_bars

        # ── Phase 1: Wait for beat detection ──────────────────────────────
        log.info("Waiting for tempo detection (need ~6 onsets)…")
        self.client.send_message("/gen/status", ["detecting tempo…"])

        while self._running:
            time.sleep(0.5)
            self.beat_tracker.update()
            if self.beat_tracker.ready:
                break

        if not self._running:
            return

        bar_dur = self.beat_tracker.bar_duration
        bpm = self.beat_tracker.bpm
        log.info("Tempo detected: %.1f BPM (bar=%.2fs)", bpm, bar_dur)
        self.client.send_message("/gen/status", [f"tempo: {bpm:.0f} BPM"])

        # ── Phase 2: Priming — user plays accompaniment as context ─────
        self._priming = True
        priming_dur = self.priming_bars * bar_dur
        log.info("PRIMING: play chords/accompaniment for %d bars (%.1fs). "
                 "Send /control/prime_done when ready, or wait for auto-switch.",
                 self.priming_bars, priming_dur)
        self.client.send_message("/gen/status", [f"priming: play chords for {self.priming_bars} bars"])

        priming_start = self.buffer.elapsed()
        while self._running and self._priming:
            if self.buffer.elapsed() - priming_start >= priming_dur:
                self._priming = False
                log.info("Priming auto-ended after %d bars", self.priming_bars)
            else:
                time.sleep(0.2)

        if not self._running:
            return

        self.client.send_message("/gen/status", ["comping: play melody"])

        gen_end_session = 0.0  # no accompaniment generated yet

        # ── Phase 3: Pipeline generation loop ────────────────────────────
        while self._running:
            # Re-estimate tempo each cycle (micro-tempo tracking)
            bpm = self.beat_tracker.update()
            bar_dur = self.beat_tracker.bar_duration
            gen_dur = self.gen_bars * bar_dur
            shift_dur = shift_bars * bar_dur

            # Determine generation window in session time
            if gen_end_session == 0.0:
                # First generation: output starts at shift_bars * bar_dur
                gen_start_session = shift_bars * bar_dur
            else:
                gen_start_session = gen_end_session

            gen_end_session_new = gen_start_session + gen_dur

            # Skip ahead if we've fallen behind (gen window already in the past)
            elapsed = self.buffer.elapsed()
            if gen_end_session_new < elapsed - gen_dur:
                log.warning("  Skipping window — fell behind (gen_end=%.1f, elapsed=%.1f)",
                            gen_end_session_new, elapsed)
                gen_end_session = gen_end_session_new
                window_num += 1
                continue

            # Wait until trigger: start computing ~latency before gen window plays
            estimated_latency = max(self._last_gen_time, self.latency_bars * bar_dur)
            trigger_time = gen_start_session - estimated_latency
            while self._running and self.buffer.elapsed() < trigger_time:
                time.sleep(0.1)

            if not self._running:
                break

            # ── Controls source: recent user bars for melody controls ─────
            ctrl_src_start = max(0.0, gen_start_session - shift_dur)
            ctrl_src_end = ctrl_src_start + gen_dur
            ctrl_user_notes = self.buffer.collect_window(ctrl_src_start, ctrl_src_end)

            # ── History: user notes as piano(0) + model output ─────────────
            hist_limit = max(0.0, gen_start_session - self.history_bars * bar_dur)
            user_raw = self.buffer.collect_window(hist_limit, max(hist_limit, ctrl_src_end))
            user_hist = [(t, dur, p, 0) for (t, dur, p, _) in user_raw]
            model_hist = self.gen_history.collect_window(hist_limit, gen_start_session)

            hist_start = hist_limit
            actual_hist_dur = gen_start_session - hist_start

            # MAX_TIME safeguard
            max_total_sec = (MAX_TIME - 1) / TIME_RESOLUTION
            if actual_hist_dur + gen_dur + DELTA > max_total_sec:
                actual_hist_dur = max_total_sec - gen_dur - DELTA
                hist_start = gen_start_session - actual_hist_dur
                user_raw = self.buffer.collect_window(hist_start, max(hist_start, ctrl_src_end))
                user_hist = [(t, dur, p, 0) for (t, dur, p, _) in user_raw]
                model_hist = self.gen_history.collect_window(hist_start, gen_start_session)
                log.warning("Clamped history to %.1fs to stay within MAX_TIME", actual_hist_dur)

            all_events = user_hist + model_hist
            inputs = notes_to_events(all_events)

            if not inputs:
                actual_hist_dur = 0.0
                hist_start = gen_start_session

            # ── Melody controls: top note from control window ─────────────
            ctrl_melody = extract_top_melody(ctrl_user_notes)
            if ctrl_melody:
                shifted = []
                for (t, dur, p, i) in ctrl_melody:
                    new_t = actual_hist_dur + t
                    shifted.append((new_t, dur, p, i))
                controls = notes_to_controls(shifted)
            else:
                controls = []

            # ── Cap total tokens to keep generation fast ─────────────────
            # More tokens = slower generation. Cap inputs so gen stays under ~2s.
            max_input_tokens = min(200, 1023 - len(controls) - 30)
            if len(inputs) > max_input_tokens:
                # trim oldest events (each event = 3 tokens)
                trim = len(inputs) - max_input_tokens
                trim = (trim // 3 + 1) * 3  # round up to event boundary
                inputs = inputs[trim:]
                log.info("  Trimmed inputs from %d to %d tokens (context cap)",
                         len(inputs) + trim, len(inputs))

            # ── Role detection for logging ────────────────────────────────
            recent_notes = ctrl_user_notes if len(ctrl_user_notes) >= 3 else user_hist
            if recent_notes and len(recent_notes) >= 3:
                improv_score, role_label = self._detect_role(recent_notes)
                self.client.send_message("/gen/role",
                                         [role_label, float(improv_score), "anticipate"])

            # ── Log window info ───────────────────────────────────────────
            note_names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
            log.info("── WINDOW %d ─── BPM=%.1f  bar=%.2fs  shift=%d bars ──",
                     window_num, bpm, bar_dur, shift_bars)
            log.info("  History: %d events (user=%d model=%d) over %.1fs",
                     len(all_events), len(user_hist), len(model_hist), actual_hist_dur)
            log.info("  Controls: %d notes → %d tokens (gen window %.1fs–%.1fs session)",
                     len(ctrl_user_notes), len(controls) // 3,
                     gen_start_session, gen_end_session_new)
            if ctrl_user_notes:
                for (t, dur, pitch, *_) in sorted(ctrl_user_notes, key=lambda x: x[0])[:8]:
                    name = note_names[pitch % 12] + str(pitch // 12 - 1)
                    log.info("    ctrl t=%5.2fs dur=%.2fs %s", t, dur, name)
                if len(ctrl_user_notes) > 8:
                    log.info("    ... (%d more)", len(ctrl_user_notes) - 8)

            # ── Generate ──────────────────────────────────────────────────
            log.info("  generate(start=%.2f, end=%.2f) inputs=%d tokens, controls=%d tokens",
                     actual_hist_dur, actual_hist_dur + gen_dur + DELTA,
                     len(inputs), len(controls))
            if controls:
                # Log first few control token triplets for debugging
                for ci in range(0, min(len(controls), 9), 3):
                    ct = controls[ci] - ATIME_OFFSET
                    cd = controls[ci+1] - ADUR_OFFSET
                    cn = controls[ci+2] - ANOTE_OFFSET
                    log.info("    control token: time=%.2f dur=%.2f note=%d",
                             ct / TIME_RESOLUTION, cd / TIME_RESOLUTION, cn)

            t_gen_start = time.time()

            try:
                with torch.no_grad():
                    events = generate(
                        self.model,
                        start_time  = actual_hist_dur,
                        end_time    = actual_hist_dur + gen_dur,
                        inputs      = inputs,
                        controls    = controls,
                        top_p       = self.top_p,
                        temperature = self.temperature,
                    )
            except Exception as exc:
                log.exception("Generation failed: %s", exc)
                self.client.send_message("/gen/status", [f"error: {exc}"])
                time.sleep(bar_dur)
                window_num += 1
                continue

            self._last_gen_time = time.time() - t_gen_start
            gen_elapsed = self._last_gen_time

            # Clip to generation window
            events = ops.clip(events, actual_hist_dur, actual_hist_dur + gen_dur,
                              clip_duration=False, seconds=True)

            n_events = len(events) // 3
            log.info("  Generated %d events in %.2fs (%.1fx realtime)",
                     n_events, gen_elapsed, gen_dur / max(gen_elapsed, 0.01))

            # ── Decode → (optional shimonization) → schedule ─────────────
            t0 = time.time()

            decoded = decode_events(events)
            t1 = time.time()
            log.info("  pipeline  decode_events  : %5.3f ms  (%d notes)", (t1-t0)*1e3, len(decoded))

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

            # Store in generated history (session time)
            self.gen_history.add_notes(decoded, session_offset=hist_start)

            # Schedule playback: map model time to wall-clock
            # gen_start_session is when these notes should play in session time
            # play_start is the current wall clock time
            # We want notes at model time actual_hist_dur to play at wall-clock gen_start_session
            # But gen_start_session is in the future relative to elapsed
            # play_offset = wall clock time corresponding to session time gen_start_session
            now_session = self.buffer.elapsed()
            play_offset = time.time() + (gen_start_session - now_session)
            schedule = notes_to_schedule(decoded, play_offset, win_start=actual_hist_dur)

            # Cancel old playback on key change
            key_notes = ctrl_user_notes if ctrl_user_notes else user_hist
            key_changed = self._detect_key_change(key_notes) if key_notes else False
            if key_changed and self._current_playback_cancel is not None:
                log.info("Window %d: key change — cancelling old playback", window_num)
                self._current_playback_cancel.set()

            cancel_event = threading.Event()
            self._current_playback_cancel = cancel_event
            threading.Thread(
                target=self._playback_thread,
                args=(schedule, cancel_event),
                daemon=True,
            ).start()

            gen_end_session = gen_end_session_new
            window_num += 1

    def _playback_thread(self, schedule: list[tuple], cancel: threading.Event):
        log.info("Playback: sending %d OSC messages to %s:%d",
                 len(schedule), self.client._address, self.client._port)
        skipped = 0
        for (target_time, address, args) in schedule:
            if cancel.is_set():
                log.info("Playback: cancelled (key change)")
                return
            now = time.time()
            # Drop notes more than 1s in the past
            if target_time < now - 1.0:
                skipped += 1
                continue
            if target_time > now:
                if cancel.wait(timeout=target_time - now):
                    log.info("Playback: cancelled (key change)")
                    return
            log.info("  → %s %s", address, args)
            self.client.send_message(address, args)
        if skipped:
            log.info("Playback: skipped %d stale messages", skipped)
        log.info("Playback: done")

    # ── Start OSC server ──────────────────────────────────────────────────────

    def run(self):
        disp = osc_dispatcher.Dispatcher()
        disp.map("/note",                   self._on_note)
        disp.map("/control/start",          self._on_start)
        disp.map("/control/stop",           self._on_stop)
        disp.map("/control/default_bpm",    self._on_default_bpm)
        disp.map("/control/top_p",          self._on_top_p)
        disp.map("/control/temperature",    self._on_temperature)
        disp.map("/control/test",           self._on_test)
        disp.map("/control/prime_done",    self._on_prime_done)
        disp.set_default_handler(self._on_any)

        server = osc_server.ThreadingOSCUDPServer(
            (self.listen_ip, self.listen_port), disp
        )
        log.info("Sending generated notes to %s:%d",
                 self.client._address, self.client._port)
        self._startup_test()
        log.info("OSC server listening on %s:%d", self.listen_ip, self.listen_port)
        server.serve_forever()

    def _startup_test(self):
        log.info("STARTUP TEST: firing C major arpeggio to %s:%d",
                 self.client._address, self.client._port)
        for pitch in [50, 62, 74, 86]:
            self.client.send_message("/gen/noteon",  [pitch, 100, 2])
            log.info("  → /gen/noteon [pitch=%d vel=100 ch=2]", pitch)
            time.sleep(1.0)
            self.client.send_message("/gen/noteoff", [pitch, 2])
        log.info("STARTUP TEST done – if Max heard 4 notes the return path is working")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    model_path = '/data/AMTmodel/pop909_10epfinal'
    client_ip = "192.168.1.2"
    listen_ip = "192.168.1.10"
    client_port = 9001
    listen_port = 9000

    server = JamServer(
        model_path          = model_path,
        listen_ip           = listen_ip,
        listen_port         = listen_port,
        client_ip           = client_ip,
        client_port         = client_port,
        gen_bars            = 4,
        history_bars        = 8,
        beats_per_bar       = 4,
        default_bpm         = 120.0,
        latency_bars        = 1,
        key_change_threshold= 0.35,
        top_p               = 0.90,
        temperature         = 0.8,
        human_instrument    = 52,   # POP909 melody = program 52 (Choir Aahs)
        shimonize           = False,
    )
    server.run()


if __name__ == "__main__":
    main()
