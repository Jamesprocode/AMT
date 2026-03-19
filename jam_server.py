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
    /control/gen_mode     string      -- "auto", "autoregress", or "anticipate"

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
import torch
from transformers import AutoModelForCausalLM
from pythonosc import dispatcher as osc_dispatcher
from pythonosc import osc_server
from pythonosc.udp_client import SimpleUDPClient

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


# ── Token helpers ─────────────────────────────────────────────────────────────

def notes_to_events(notes: list[tuple]) -> list[int]:
    """Convert (t_rel, dur, pitch, instrument) list → regular AMT event tokens.

    Passed as inputs= so the model treats them as already-happened music
    and generates a continuation (AUTOREGRESS mode).
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


def notes_to_controls(notes: list[tuple]) -> list[int]:
    """Convert (t_rel, dur, pitch, instrument) list → anticipatory control tokens.

    Each note is shifted forward by DELTA seconds so the model sees them as
    future constraints and generates accompaniment for the same [0, window_size].
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
        generation_length: float = 15.0,
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
        self.window_size           = window_size
        self.generation_length     = generation_length
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
        self._prev_pitch_classes: set  = set()
        self._current_playback_cancel: threading.Event | None = None

        # ── Role detection & mode switching state ─────────────────────────
        self._current_role       = "improv"      # detected user role
        self._role_score_ema     = 0.5           # EMA of improv_score
        self._role_ema_alpha     = 0.4           # smoothing factor
        self._role_switch_high   = 0.65          # EMA above → user is improvising
        self._role_switch_low    = 0.35          # EMA below → user is accompanying
        self._generation_mode    = "anticipate"  # "autoregress" or "anticipate"
        self._phrases_in_mode    = 0             # count since last mode switch
        self._min_phrases_switch = 2             # minimum before allowing switch
        self._looped_controls: list[int] = []    # cached control tokens for continuity
        self._mode_override      = "auto"        # OSC manual override

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
        self.client.send_message("/gen/status", ["session started"])
        log.info("Session started  window=%.1fs  top_p=%.2f  temp=%.2f",
                 self.window_size, self.top_p, self.temperature)

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

    def _on_window_size(self, address, value):
        self.window_size = float(value)
        log.info("window_size → %.2f s", self.window_size)

    def _on_top_p(self, address, value):
        self.top_p = float(value)
        log.info("top_p → %.3f", self.top_p)

    def _on_temperature(self, address, value):
        self.temperature = float(value)
        log.info("temperature → %.3f", self.temperature)

    def _on_gen_mode(self, address, value):
        value = str(value).strip().lower()
        if value in ("auto", "autoregress", "anticipate"):
            self._mode_override = value
            log.info("gen_mode override → %s", value)
        else:
            log.warning("Unknown gen_mode: %s (use auto/autoregress/anticipate)", value)

    # ── Key change detection ──────────────────────────────────────────────────

    def _detect_key_change(self, notes: list[tuple]) -> bool:
        """Return True if the pitch class content has shifted dramatically.

        Uses Jaccard similarity between current and previous window pitch classes.
        A similarity below key_change_threshold signals a key/tonality change.
        """
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

    # ── Role detection ────────────────────────────────────────────────────────

    def _repetition_score(self, notes: list[tuple]) -> float:
        """Detect repetition within the current window.

        Tries motif lengths from 2 to len/2 and checks how well the pitch
        sequence matches when shifted by that period. Returns [0, 1] where
        1 = perfectly periodic.
        """
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

        Returns (improv_score, role_label) where improv_score is 0.0–1.0
        and role_label is "improv" or "accomp".
        """
        if len(notes) < 3:
            return (0.5, self._current_role)

        sorted_notes = sorted(notes, key=lambda x: x[0])

        # Signal 1: Polyphony — group notes by onset (30ms tolerance)
        onset_groups = []
        current_group = [sorted_notes[0]]
        for note in sorted_notes[1:]:
            if note[0] - current_group[-1][0] < 0.03:
                current_group.append(note)
            else:
                onset_groups.append(current_group)
                current_group = [note]
        onset_groups.append(current_group)

        # If all onsets are single notes → monophonic → user is improvising
        max_group = max(len(g) for g in onset_groups)
        if max_group == 1:
            log.info("Role detection: monophonic input → improv")
            return (1.0, "improv")

        avg_sim = sum(len(g) for g in onset_groups) / len(onset_groups)
        polyphony_norm = max(0.0, min(1.0, (avg_sim - 1.0) / 2.5))
        improv_from_polyphony = 1.0 - polyphony_norm

        # Signal 2: Repetition (LCS vs previous phrase)
        rep_score = self._repetition_score(notes)
        improv_from_repetition = 1.0 - rep_score

        # Signal 3: Rhythmic regularity (IOI coefficient of variation)
        # Use onset group start times for IOI (skip same-onset notes)
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

        # Signal 5: Stepwise motion (fraction of sequential intervals <= 2 semitones)
        sequential_pitches = [g[0][2] for g in onset_groups]  # first note of each onset group
        if len(sequential_pitches) >= 2:
            intervals = [abs(sequential_pitches[i + 1] - sequential_pitches[i])
                         for i in range(len(sequential_pitches) - 1)]
            stepwise_count = sum(1 for iv in intervals if iv <= 2)
            improv_from_intervals = stepwise_count / len(intervals)
        else:
            improv_from_intervals = 0.5

        # Combined score
        improv_score = (0.30 * improv_from_polyphony +
                        0.20 * improv_from_repetition +
                        0.20 * improv_from_rhythm +
                        0.15 * improv_from_range +
                        0.15 * improv_from_intervals)

        role_label = "improv" if improv_score > 0.5 else "accomp"

        log.info("Role detection: score=%.2f [poly=%.2f rep=%.2f rhythm=%.2f range=%.2f step=%.2f] → %s",
                 improv_score, improv_from_polyphony, improv_from_repetition,
                 improv_from_rhythm, improv_from_range, improv_from_intervals, role_label)

        return (improv_score, role_label)

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
            self._looped_controls = []  # invalidate cache on switch
            log.info("Mode switch → ANTICIPATE (user improvising, model accompanies)")
        elif self._current_role == "improv" and self._role_score_ema < self._role_switch_low:
            self._current_role = "accomp"
            self._generation_mode = "autoregress"
            self._phrases_in_mode = 0
            self._looped_controls = []
            log.info("Mode switch → AUTOREGRESS (user accompanying, model solos)")

        return self._generation_mode

    # ── Control looping ───────────────────────────────────────────────────────

    def _loop_controls_into_window(self, notes: list[tuple], window_dur: float, instrument: int | None = None) -> list[int]:
        """Repeat user melody pattern cyclically to fill [0, window_dur], return as control tokens.

        If instrument is specified, override the instrument on all notes.
        """
        if not notes:
            return []
        min_t = min(n[0] for n in notes)
        max_t = max(n[0] + n[1] for n in notes)
        pattern_dur = max(max_t - min_t, 0.5)
        looped = []
        offset = 0.0
        while offset < window_dur:
            for (t, dur, pitch, instr) in notes:
                new_t = (t - min_t) + offset
                if new_t < window_dur:
                    looped.append((new_t, dur, pitch, instrument if instrument is not None else instr))
            offset += pattern_dur
        return notes_to_controls(looped)

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

            key_changed = self._detect_key_change(notes)

            # ── Detect role & decide generation mode ──────────────────────
            improv_score, role_label = self._detect_role(notes)
            gen_mode = self._update_role(improv_score)

            if self._mode_override != "auto":
                gen_mode = self._mode_override

            self.client.send_message("/gen/role",
                                     [role_label, float(improv_score), gen_mode])

            log.info("Window %d: %d notes  mode=%s  ema=%.2f – generating %.1fs …",
                     window_num, len(notes), gen_mode, self._role_score_ema,
                     self.generation_length)
            t_gen_start = time.time()

            try:
                note_names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
                log.info("── PHRASE ────────────────────────────────────────")
                for (t, dur, pitch, instr) in sorted(notes, key=lambda x: x[0]):
                    name = note_names[pitch % 12] + str(pitch // 12 - 1)
                    log.info("  t=%5.2fs  dur=%.2fs  %s (pitch=%d  instr=%d)",
                             t, dur, name, pitch, instr)
                log.info("─────────────────────────────────────────────────")

                # Always anticipate: user input as controls, model generates the other role
                # accomp (comping) → controls as piano (0), model generates melody
                # improv (melody)  → controls as voice (52), model generates chords
                control_instr = 0 if role_label == "accomp" else 52

                self._looped_controls = self._loop_controls_into_window(
                    notes, self.generation_length + DELTA, instrument=control_instr)
                log.info("  Controls: %d tokens (looped, instr=%d)",
                         len(self._looped_controls), control_instr)

                with torch.no_grad():
                    events = generate(
                        self.model,
                        start_time  = 0,
                        end_time    = self.generation_length + DELTA,
                        inputs      = [],
                        controls    = self._looped_controls,
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

            # clip to playback window (discard events beyond generation_length)
            events = ops.clip(events, 0, self.generation_length,
                              clip_duration=False, seconds=True)

            # decode → (optional shimonization) → schedule
            play_start = time.time()
            t0 = time.time()

            decoded = decode_events(events)
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

            schedule = notes_to_schedule(decoded, play_start, 0.0)
            t7 = time.time(); log.info("  pipeline  notes_to_sched : %5.3f ms", (t7-t1)*1e3)
            # cancel old playback now that new one is ready — no silence gap
            if key_changed and self._current_playback_cancel is not None:
                log.info("Window %d: key change — cancelling old playback now", window_num)
                self._current_playback_cancel.set()
            cancel_event = threading.Event()
            self._current_playback_cancel = cancel_event
            threading.Thread(
                target=self._playback_thread,
                args=(schedule, cancel_event),
                daemon=True,
            ).start()

            window_num += 1

    def _playback_thread(self, schedule: list[tuple], cancel: threading.Event):
        log.info("Playback: sending %d OSC messages to %s:%d",
                 len(schedule), self.client._address, self.client._port)
        for (target_time, address, args) in schedule:
            if cancel.is_set():
                log.info("Playback: cancelled (key change)")
                return
            now = time.time()
            if target_time > now:
                # sleep in small steps so cancel is checked promptly
                if cancel.wait(timeout=target_time - now):
                    log.info("Playback: cancelled (key change)")
                    return
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
        disp.map("/control/test",           self._on_test)
        disp.map("/control/gen_mode",       self._on_gen_mode)
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
    #                     help="Sampling temperature (default 1.0)")
    # args = parser.parse_args()

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
        window_size         = 5.0,
        generation_length   = 5.0,
        key_change_threshold= 0.35,
        top_p               = 0.90,
        temperature         = 0.8,
        shimonize           = False,
    )
    server.run()


if __name__ == "__main__":
    main()
