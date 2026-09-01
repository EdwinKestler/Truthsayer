"""HUD drawing for pulse waveform, magnified face, voice, VRMS, cue load."""
from __future__ import annotations

import cv2
import numpy as np


def _put(image, text, x, y, scale=0.55, color=(255, 255, 255)):
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_waveform(image, waveform, y0=None):
    if waveform is None or len(waveform) < 4:
        return
    h, w = image.shape[:2]
    box_h = max(36, h // 14)
    box_w = min(w - 20, 420)
    x0, y0 = 10, (h - box_h - 10) if y0 is None else y0
    roi = image[y0:y0 + box_h, x0:x0 + box_w]
    if roi.size == 0:
        return
    overlay = roi.copy()
    cv2.rectangle(overlay, (0, 0), (box_w - 1, box_h - 1), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.45, roi, 0.55, 0, roi)

    sig = np.asarray(waveform, dtype=np.float64)
    if not np.any(np.isfinite(sig)):
        return
    sig = np.nan_to_num(sig, nan=0.0)
    mx = np.max(np.abs(sig)) or 1.0
    sig = sig / mx
    xs = np.linspace(2, box_w - 3, num=min(sig.size, box_w - 4)).astype(np.int32)
    idx = np.linspace(0, sig.size - 1, num=xs.size).astype(np.int32)
    ys = (box_h / 2 - sig[idx] * (box_h / 2 - 4)).astype(np.int32)
    pts = np.stack([xs, np.clip(ys, 1, box_h - 2)], axis=1)
    cv2.polylines(roi, [pts], False, (80, 220, 80), 1, cv2.LINE_AA)
    _put(image, "rPPG pulse", x0 + 4, y0 - 6, 0.45, (180, 255, 180))


def draw_magnified(image, panel):
    if panel is None:
        return
    h, w = image.shape[:2]
    side = max(80, min(h // 4, w // 5))
    small = cv2.resize(panel, (side, side), interpolation=cv2.INTER_LINEAR)
    y0, x0 = h - side - 10, w - side - 10
    if y0 < 0 or x0 < 0:
        return
    image[y0:y0 + side, x0:x0 + side] = small
    _put(image, "motion x18", x0, y0 - 6, 0.45, (200, 200, 255))


def draw_cue_bar(image, score, label):
    h, w = image.shape[:2]
    bar_w = min(w // 3, 360)
    bar_h = 12
    x0 = (w - bar_w) // 2
    y0 = 8
    cv2.rectangle(image, (x0, y0), (x0 + bar_w, y0 + bar_h), (40, 40, 40), -1)
    fill = int(bar_w * max(0.0, min(1.0, score)))
    color = (80, 200, 80) if score < 0.25 else ((40, 180, 220) if score < 0.55 else (40, 40, 220))
    cv2.rectangle(image, (x0, y0), (x0 + fill, y0 + bar_h), color, -1)
    cv2.rectangle(image, (x0, y0), (x0 + bar_w, y0 + bar_h), (0, 0, 0), 1)
    _put(image, "cue load {:.0f}%  {}".format(score * 100, label), x0, y0 + bar_h + 16, 0.5, color)


def draw_metrics(image, rppg, vrms, voice, calibrated, baseline=None):
    h, w = image.shape[:2]
    x = int(w * 0.62)
    y = 50
    bpm = rppg.get("bpm") or 0
    snr = rppg.get("snr") or 0
    bpm_txt = "POS BPM: ..." if bpm < 40 else "POS BPM: {:.0f}  snr {:.2f}".format(bpm, snr)
    _put(image, bpm_txt, x, y)
    y += 22
    _put(image, "VRMS: {:.4f}  x{:.1f}".format(vrms.get("vrms", 0.0), vrms.get("ratio", 1.0)), x, y)
    y += 22
    if voice:
        if voice.get("voiced"):
            _put(image, "Pitch: {:.0f} Hz  {:+.0f}".format(voice.get("pitch", 0), voice.get("pitch_delta", 0)), x, y, color=(180, 220, 255))
            y += 22
            _put(image, "Voice RMS: {:.0f} dB  peak {:.0f} Hz".format(voice.get("rms", -80), voice.get("peak_hz", 0)), x, y, color=(180, 220, 255))
        else:
            _put(image, "Voice: {}".format(voice.get("status", "silent")), x, y, 0.5, (160, 160, 160))
    if not calibrated and not (baseline and baseline.get("phase") in ("cal_true", "cal_lie")):
        _put(image, "calibrating baseline...", 10, h - 16, 0.5, (180, 180, 180))


def draw_reason(image, agent):
    if not agent:
        return
    h, w = image.shape[:2]
    text = agent.get("explanation") or ""
    if agent.get("review"):
        text = "REVIEW  " + text
    if not text:
        return
    y = 28
    color = (40, 40, 220) if agent.get("review") else (200, 200, 200)
    max_chars = max(40, w // 12)
    line = text[:max_chars]
    _put(image, line, 10, y, 0.42, color)


def draw_utterance(image, utt):
    if not utt:
        return
    h, _w = image.shape[:2]
    hud = utt.get("hud") or ""
    if not hud:
        return
    if not utt.get("closed"):
        hud = "... " + hud
    color = (180, 220, 180) if utt.get("closed") else (160, 160, 160)
    _put(image, hud, 10, h - 18, 0.42, color)


def draw_cal(image, baseline):
    if not baseline:
        return
    phase = baseline.get("phase")
    rem = baseline.get("remaining") or 0
    prompt = baseline.get("prompt") or ""
    if phase == "cal_true":
        _put(image, "CAL TRUE  {:.0f}s   N=skip  Enter=next".format(rem), 10, 48, 0.5, (80, 220, 80))
        _put(image, prompt, 10, 72, 0.45, (200, 255, 200))
        _put(image, "truth-talk samples {}".format(baseline.get("n_true") or 0), 10, 94, 0.4, (160, 160, 160))
    elif phase == "cal_lie":
        _put(image, "CAL LIE  invent FALSE answers  {:.0f}s   N=skip  Enter=next".format(rem), 10, 48, 0.5, (40, 180, 255))
        _put(image, prompt, 10, 72, 0.45, (200, 230, 255))
        _put(image, "directed-fab samples {}".format(baseline.get("n_fab") or 0), 10, 94, 0.4, (160, 160, 160))
    elif phase == "warmup":
        _put(image, "warmup {}/{}".format(baseline.get("warmup_n") or 0, baseline.get("warmup_need") or 0),
             10, 48, 0.5, (180, 180, 180))


def draw_hud(image, rppg, vrms, voice, fusion, panel=None, calibrated=False, agent=None, utterance=None, baseline=None):
    draw_cue_bar(image, fusion.get("score", 0.0), fusion.get("label", ""))
    if baseline and baseline.get("phase") in ("cal_true", "cal_lie", "warmup") and not calibrated:
        draw_cal(image, baseline)
    else:
        draw_reason(image, agent)
    draw_waveform(image, rppg.get("waveform"))
    draw_magnified(image, panel)
    draw_metrics(image, rppg, vrms, voice or {}, calibrated, baseline)
    if calibrated:
        draw_utterance(image, utterance)
        score_hud = ((baseline or {}).get("score") or {}).get("hud")
        if score_hud:
            h, w = image.shape[:2]
            _put(image, score_hud, 10, h - 38, 0.42, (180, 220, 255))
