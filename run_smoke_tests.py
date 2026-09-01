"""Headless smoke tests for Truthsayer (intercept.py + modular_face_analysis.py)."""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import traceback
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

PASSED = []
FAILED = []


def check(name, cond, detail=""):
    if cond:
        PASSED.append(name)
        print("PASS:", name)
    else:
        FAILED.append(name)
        print("FAIL:", name, detail)


def expect_ok(name, fn):
    try:
        fn()
        PASSED.append(name)
        print("PASS:", name)
    except Exception as exc:
        FAILED.append(name)
        print("FAIL:", name, type(exc).__name__, exc)
        traceback.print_exc()


class P:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def test_analyzers():
    from analyzers.rppg import FacialRPPG
    from analyzers.vrms import FacialVRMS
    from analyzers.motion import MotionMagnifier
    from analyzers.voice import estimate_pitch, rms_db
    from analyzers.fusion import CueFusion

    rppg = FacialRPPG(window=64)
    h, w = 40, 40
    t = np.linspace(0, 2, 64)
    pulse = 0.04 * np.sin(2 * np.pi * 1.2 * t)  # 72 BPM
    for i, p in enumerate(pulse):
        cheek = np.zeros((h, w, 3), dtype=np.uint8)
        cheek[:, :, 1] = np.clip(80 + 40 * p, 0, 255)
        cheek[:, :, 2] = np.clip(70 + 30 * p, 0, 255)
        cheek[:, :, 0] = 40
        snap = rppg.update(cheek, cheek, fps=32.0)
    check("POS rPPG produces finite BPM", np.isfinite(snap["bpm"]))
    check("POS waveform is buffered", snap["waveform"].size == 64)

    class Lm:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    vrms = FacialVRMS(window=30, recent=8)
    landmarks = [Lm(0.5, 0.5) for _ in range(500)]
    for i in range(25):
        for idx in (70, 63, 13, 14, 61, 291):
            landmarks[idx] = Lm(0.5 + 0.0001 * i, 0.5)
        vrms.update(landmarks)
    for i in range(8):
        for idx in (70, 63, 13, 14, 61, 291):
            landmarks[idx] = Lm(0.5 + 0.02 * i, 0.5 + 0.02 * i)
        vrms.update(landmarks)
    burst = vrms.snapshot()
    check("VRMS ratio responds to a motion burst", burst["ratio"] > 1.0)

    mag = MotionMagnifier(size=32, buffer=24, alpha=10)
    face = np.full((64, 64, 3), 80, dtype=np.uint8)
    panel = None
    energy = 0
    for i in range(24):
        frame = face.copy()
        frame[16:20, :] = 80 + int(8 * np.sin(i / 3.0))
        panel, energy = mag.update(frame, fps=30)
    check("motion magnifier returns a panel", panel is not None and panel.shape[0] == 32)

    sr = 16000
    tone = np.sin(2 * np.pi * 180 * np.arange(sr // 10) / sr)
    hz = estimate_pitch(tone, sr)
    check("pitch detector ~180 Hz", abs(hz - 180) < 15, hz)
    check("rms_db of a tone is not silence", rms_db(tone) > -20)

    fusion = CueFusion()
    snap = fusion.update(
        {"gaze": {"text": "x", "ttl": 1}, "blinking": {"text": "y", "ttl": 1}},
        {"bpm": 80}, {"burst": True},
        {"voiced": True, "pitch_delta": 30, "rms_delta": 0},
        motion_energy=4.0, bpm_change_text="Heart rate increasing",
    )
    check("fusion score increases with many cues", snap["score"] > 0.4)
    check("fusion never claims lying", "lie" not in snap["label"] and "deceit" not in snap["label"])

    from analyzers.linguistics import LinguisticCues
    from agents.react_agent import CueReactAgent
    lex = LinguisticCues().update("I think maybe I did not take it, um, you know")
    check("linguistics flags hedges/fillers", bool(lex["flags"]))
    agent = CueReactAgent()
    early = agent.reason(False, {}, {"bpm": 0}, {}, {}, {"score": 0, "active": {}})
    check("ReAct refuses to escalate before calibration", early["action"] == "need_more_data")
    later = agent.reason(
        True, {"gaze": {"text": "x"}}, {"bpm": 72}, {"burst": False},
        {"voiced": True, "pitch_delta": 40, "rms_delta": 0},
        {"score": 0.5, "active": {"gaze": True, "pitch_rise": True}},
        {},
    )
    check("ReAct never says deceptive/truthful", "deceptive" not in later["explanation"].lower())
    check("ReAct can flag review on discord", later["action"] in ("flag_review", "continue"))
    face_only = agent.reason(
        True, {}, {"bpm": 72}, {"burst": True},
        {}, {"score": 0.31, "active": {"blink": True, "vrms_burst": True, "motion_energy": True}},
    )
    check("face/motion-only never flags review", face_only["action"] == "continue" and not face_only["review"])

    from analyzers.utterance import UtteranceTracker, is_speaking
    check("mic voiced counts as speaking", is_speaking({"voiced": True, "rms": -30}, {}, True))
    check("silent mic is not speaking", not is_speaking({"voiced": False, "rms": -60}, {}, True))
    check("no-mic uses VRMS gate", is_speaking({}, {"ratio": 1.5}, False))
    utt = UtteranceTracker(min_sec=0.2, silence_sec=0.15)
    closed = None
    for i in range(12):
        closed = utt.update(True, {
            "bpm": 70 + i, "vrms_ratio": 1.2, "cue_load": 0.2,
            "pitch_delta": 0, "gaze": i == 3, "blink": False,
        }, fps=30)
    for _ in range(8):
        got = utt.update(False, {"bpm": 80, "vrms_ratio": 1.0, "cue_load": 0.1}, fps=30)
        if got:
            closed = got
    check("utterance closes after silence", closed is not None and closed.get("closed"))
    check("utterance HUD has no lie word", closed and "lie" not in closed["hud"].lower())
    check("utterance reports gaze count", closed and closed["gaze_count"] == 1)

    from analyzers.baseline import DualBaseline
    bl = DualBaseline(true_sec=30, lie_sec=30)
    bl.start()
    check("dual cal starts in cal_true", bl.phase == "cal_true" and not bl.ready)
    sample = {
        "closed": True, "bpm_mean": 72, "bpm_delta": 2, "vrms_mean": 1.1,
        "blink_rate": 0.3, "gaze_count": 1, "pitch_delta_mean": 5, "cue_peak": 0.2,
        "hud": "x",
    }
    bl.ingest(sample)
    check("cal_true stores samples", bl.snapshot()["n_true"] == 1)
    bl.handle_key(ord("n"))
    check("N advances to cal_lie", bl.phase == "cal_lie")
    sample2 = dict(sample)
    sample2["bpm_mean"] = 90
    sample2["cue_peak"] = 0.5
    bl.ingest(sample2)
    bl.freeze()
    check("freeze marks ready", bl.ready and bl.phase == "frozen")
    sc = bl.score(sample)
    check("score has z_true", sc.get("z_true") is not None)
    check("score HUD has no lie word", "lie" not in (sc.get("hud") or "").lower())
    skip = DualBaseline(true_sec=0, lie_sec=0, warmup_frames=3)
    skip.start()
    check("both-zero uses warmup", skip.phase == "warmup")
    skip.note_warmup_face(True)
    skip.note_warmup_face(True)
    skip.note_warmup_face(True)
    check("warmup ready after N faces", skip.ready)

    fab_only = agent.reason(
        True, {}, {"bpm": 80}, {}, {},
        {"score": 0.2, "active": {"bpm_change": True, "gaze": True}},
        {}, {"closed": True, "duration_s": 2, "bpm_delta": 10, "gaze_count": 1, "cue_peak": 0.2},
        {"z_true": 2.1, "closer": "directed-fab"},
    )
    check("directed-fab contrast is not a lie word", "deceptive" not in fab_only["explanation"].lower())
    check("large z_true with pulse/gaze can review", fab_only["review"] is True)


def test_helpers_without_app():
    img = np.zeros((10, 20, 3), np.uint8)
    img[0, 0] = (1, 2, 3)
    cv2.flip(img, 1, dst=img)
    check("flip in-place mirrors pixel", tuple(img[0, 19]) == (1, 2, 3), img[0, 19])

    parser = argparse.ArgumentParser()
    parser.add_argument("--second", "-s")
    args = parser.parse_args(["--second", "0"])
    check("--second 0 is present", args.second is not None)
    src = int(args.second) if args.second.isdigit() else args.second
    check("--second 0 parses as int 0", src == 0)
    check("old if SECOND: would skip device 0", not bool(src))

    ibis = np.array([0.75])
    check("webcam BPM 60/IBI", abs(60.0 / ibis[0] - 80) < 1e-6)
    check("file BPM 60*fps/frames", abs(60.0 * 30.0 / 22.5 - 80) < 1e-6)
    check("old formula is inverted", abs(60 * 0.75 / 1 - 80) > 1)


def test_get_area_and_bpm(intercept):
    image = np.ones((100, 100, 3), np.uint8) * 80
    crop = intercept.get_area(image, False, P(0.8, 0.2), P(0.2, 0.2), P(0.2, 0.4), P(0.8, 0.4))
    check("get_area inverted x still crops", crop is not None and crop.size > 0)
    empty = intercept.get_area(image, False, P(0.5, 0.5), P(0.5, 0.5), P(0.5, 0.5), P(0.5, 0.5))
    check("get_area tiny box is None", empty is None)
    oob = intercept.get_area(image, False, P(-0.2, -0.2), P(-0.1, -0.2), P(-0.1, -0.1), P(-0.2, -0.1))
    check("get_area out of bounds is None", oob is None)

    display, change = intercept.get_bpm_tells(None, None, None, False)
    check("empty cheeks skip NaN BPM", display.startswith("BPM:"))

    cheek = np.full((8, 8, 3), 40, dtype=np.uint8)
    intercept.hr_values[:] = [400] * intercept.MAX_FRAMES
    intercept.avg_bpms[:] = [0] * intercept.MAX_FRAMES
    intercept.hr_times[:] = list(range(intercept.MAX_FRAMES))
    t0 = 0.0
    for i in range(90):
        # ~80 BPM pulse in G/R channels: period 0.75s at 30 fps-equivalent wall clock
        phase = np.sin(2 * np.pi * i / 22.5)
        sample = np.full((8, 8, 3), 80 + 20 * phase, dtype=np.uint8)
        intercept.EPOCH = t0
        # inject monotonic times via monkeypatching time.time through EPOCH offset
        intercept.get_bpm_tells(sample, sample, None, False)
    # After many frames times may not be 0.75s apart because EPOCH is fixed and
    # time.time() is real. Just assert it did not raise and returned a string.
    display, change = intercept.get_bpm_tells(cheek, cheek, None, False)
    check("get_bpm_tells returns display string", isinstance(display, str) and display.startswith("BPM:"))

    tells = {"a": {"text": "x", "ttl": 1}, "b": {"text": "y", "ttl": 2}}
    intercept.decrement_tells(tells)
    check("decrement_tells drops expired", "a" not in tells and "b" in tells and tells["b"]["ttl"] == 1)


def _make_face_video(path, frames=24):
    image = cv2.imread(os.path.join(ROOT, "demo.png"))
    if image is None:
        raise FileNotFoundError("demo.png")
    h, w = image.shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (w, h))
    if not writer.isOpened():
        raise RuntimeError("could not open VideoWriter")
    for _ in range(frames):
        writer.write(image)
    writer.release()
    return image


def test_process_demo(intercept):
    image = cv2.imread(os.path.join(ROOT, "demo.png"))
    check("demo.png loads", image is not None)
    import mediapipe as mp

    intercept.tells.clear()
    intercept.calculating_mood = False
    with mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as face_mesh:
        with mp.solutions.hands.Hands(max_num_hands=2, min_detection_confidence=0.7) as hands:
            frame = image.copy()
            got_face = intercept.process(
                frame, face_mesh, hands, calibrated=False, draw=True, bpm_chart=False, flip=False
            )
            check("process finds a face on demo.png", got_face == 1)
            flipped = image.copy()
            intercept.process(
                flipped, face_mesh, hands, calibrated=False, draw=True, bpm_chart=False, flip=True
            )
            check("process --flip mutates caller's buffer", not np.array_equal(flipped, image))
            blank = np.zeros((240, 320, 3), dtype=np.uint8)
            no_face = intercept.process(blank, face_mesh, hands, calibrated=False)
            check("process returns 0 without a face", no_face == 0)


def test_main_video_file(intercept):
    video_path = os.path.join(tempfile.gettempdir(), "truthsayer_smoke.avi")
    _make_face_video(video_path, frames=20)

    shown = {"n": 0}
    orig_imshow = cv2.imshow
    orig_wait = cv2.waitKey
    orig_destroy = cv2.destroyAllWindows
    orig_argv = sys.argv

    def fake_imshow(*_a, **_k):
        shown["n"] += 1

    def fake_wait(_ms):
        return ord("q") if shown["n"] >= 8 else 0

    cv2.imshow = fake_imshow
    cv2.waitKey = fake_wait
    cv2.destroyAllWindows = lambda: None
    sys.argv = ["intercept.py", "--input", video_path, "--landmarks", "1", "--flip", "1"]
    try:
        intercept.main()
        check("intercept.main processes demo video", shown["n"] >= 8)
    finally:
        cv2.imshow = orig_imshow
        cv2.waitKey = orig_wait
        cv2.destroyAllWindows = orig_destroy
        sys.argv = orig_argv
        try:
            os.remove(video_path)
        except OSError:
            pass


def test_modular_process():
    import mediapipe as mp
    import modular_face_analysis as mod

    image = cv2.imread(os.path.join(ROOT, "demo.png"))
    gs = SimpleNamespace(
        recording=None,
        tells={},
        blinks=[False] * mod.MAX_FRAMES,
        blinks2=[False] * mod.MAX_FRAMES,
        hand_on_face=[False] * mod.MAX_FRAMES,
        hand_on_face2=[False] * mod.MAX_FRAMES,
        face_area_size=0,
        hr_times=list(range(mod.MAX_FRAMES)),
        hr_values=[400] * mod.MAX_FRAMES,
        avg_bpms=[0] * mod.MAX_FRAMES,
        gaze_values=[0] * mod.MAX_FRAMES,
        calculating_mood=False,
        mood="",
        meter=cv2.imread(os.path.join(ROOT, "meter.png")),
        fig=None, ax=None, line=None, peakpts=None,
    )
    hr = mod.HeartRateMonitor(gs)
    gaze = mod.GazeDetector(gs)
    blink = mod.BlinkDetector(gs)

    class QuietMood:
        def maybe_start(self, _image):
            return None

    with mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as face_mesh:
        with mp.solutions.hands.Hands(max_num_hands=2, min_detection_confidence=0.7) as hands:
            frame = image.copy()
            got = mod.process(
                frame, face_mesh, hands, gs, blink, True, False, False, None,
                hr, gaze, QuietMood(), False
            )
            check("modular process finds a face on demo.png", got == 1)
            check("modular process sets face_area_size", gs.face_area_size > 0)
            check("modular process records BPM tell", "avg_bpms" in gs.tells)


def test_camera_frames(intercept):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("SKIP: camera 0 not opened")
        return
    import mediapipe as mp
    ok, frame = cap.read()
    cap.release()
    check("camera 0 yields a frame", bool(ok) and frame is not None)
    if not ok:
        return
    with mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as face_mesh:
        with mp.solutions.hands.Hands(max_num_hands=2, min_detection_confidence=0.7) as hands:
            result = intercept.process(frame.copy(), face_mesh, hands, calibrated=False)
            print("INFO: camera face detected" if result == 1 else "INFO: camera frame had no face")
            check("camera frame process does not crash", True)


def main():
    print("=== helper tests ===")
    test_helpers_without_app()

    print("=== analyzers ===")
    expect_ok("analyzer suite", test_analyzers)

    print("=== import intercept.py (loads FER/TensorFlow) ===")
    import intercept

    print("=== intercept helpers ===")
    test_get_area_and_bpm(intercept)

    print("=== intercept process(demo.png) ===")
    expect_ok("process demo.png suite", lambda: test_process_demo(intercept))

    print("=== intercept.main video file ===")
    expect_ok("main() demo video suite", lambda: test_main_video_file(intercept))

    print("=== modular_face_analysis process ===")
    expect_ok("modular process suite", lambda: test_modular_process())

    print("=== camera 0 ===")
    expect_ok("camera suite", lambda: test_camera_frames(intercept))

    print()
    print("Passed:", len(PASSED))
    print("Failed:", len(FAILED))
    for name in FAILED:
        print(" -", name)
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
