"""ReAct-style reasoner over Truthsayer cues.

Pattern from rUv's Liar-Ai gist (Reason + Act traces), but the *actions*
are operational (continue / flag_review / need_more_data), never
'Deceptive' or 'Truthful'. Rules follow docs/SCIENCE.md.
"""
from __future__ import annotations


class CueReactAgent:
    def __init__(self):
        self.last = {
            "thoughts": [],
            "action": "need_more_data",
            "review": False,
            "explanation": "calibrating",
        }

    FACE_ONLY = {"blink", "lips", "hand", "vrms_burst", "motion_energy"}
    AUDIO_ONLY = {"pitch_rise", "voice_energy"}
    MOTION_ONLY = {"vrms_burst", "motion_energy"}

    def reason(self, calibrated, tells, rppg, vrms, voice, fusion, linguistics=None, utterance=None, baseline_score=None):
        thoughts = []
        if not calibrated:
            self.last = {
                "thoughts": ["Thought: baseline not finished; refuse to escalate."],
                "action": "need_more_data",
                "review": False,
                "explanation": "calibrating baseline",
            }
            return self.last

        active = list((fusion or {}).get("active") or {})
        active_set = set(active)
        score = float((fusion or {}).get("score") or 0.0)
        thoughts.append("Observation: cue load {:.0f}% active={}.".format(score * 100, active or "none"))

        pulse = "bpm_change" in active_set
        gaze = "gaze" in active_set
        voice_only = bool(active_set) and active_set <= self.AUDIO_ONLY
        motion_only = bool(active_set) and active_set <= self.MOTION_ONLY
        face_only = bool(active_set) and active_set <= self.FACE_ONLY
        ling_flags = list(((linguistics or {}).get("flags") or {}).keys())
        utt = utterance or {}

        # Neuro-symbolic rules (literature-weighted, not P(lie))
        if pulse and gaze:
            thoughts.append("Rule: pulse change + gaze shift is the DDPM-supported cluster; keep, do not treat as proof.")
        elif pulse:
            thoughts.append("Rule: heart-rate change is the stronger single remote cue; still arousal/load, not veracity.")
        elif gaze:
            thoughts.append("Rule: gaze/saccade-like shift is weakly supported; insufficient alone.")

        if "vrms_burst" in active or "motion_energy" in active:
            thoughts.append("Rule: amplified facial motion / VRMS burst is inspectable, not diagnostic (DDPM micro-expression n.s.).")
        if "pitch_rise" in active or "voice_energy" in active:
            thoughts.append("Rule: pitch/RMS shift tracks arousal (Ekman); commercial VSA is chance-level (Harnsberger 2009).")
        if voice_only:
            thoughts.append("Thought: audio-only cluster — do not escalate.")
        if motion_only or face_only:
            thoughts.append(
                "Thought: face/motion-only cluster — do not escalate "
                "(GLD 2104.12345: static-face nets are ~chance out of domain)."
            )
        if utt.get("closed"):
            thoughts.append(
                "Observation: statement {:.1f}s ΔBPM {:+.0f} gaze {} cue_peak {:.2f}.".format(
                    utt.get("duration_s") or 0.0,
                    utt.get("bpm_delta") or 0.0,
                    utt.get("gaze_count") or 0,
                    utt.get("cue_peak") or 0.0,
                )
            )
            if (utt.get("bpm_delta") or 0) > 8 and (utt.get("gaze_count") or 0) >= 1:
                thoughts.append("Rule: this *utterance* shows pulse+gaze together; still not a lie label.")

        b = baseline_score or {}
        z_true = b.get("z_true")
        closer = b.get("closer")
        if z_true is not None:
            thoughts.append("Observation: vs truth-talk z={:.1f} closer={}.".format(z_true, closer or "?"))
        if closer == "directed-fab":
            thoughts.append("Thought: resembles directed-fabrication template; low-stakes, not a verdict.")

        if ling_flags:
            thoughts.append("Observation: lexical markers {} (Newman-style; small effects).".format(ling_flags))
            mood = ""
            # incongruence is noted if caller stuffed mood into linguistics
            if linguistics.get("mood"):
                mood = str(linguistics.get("mood"))
            if mood in {"fear", "angry", "sad"} and not ling_flags:
                thoughts.append("Rule: displayed negative affect with bland wording — possible mismatch; flag review.")

        channels = 0
        channels += 1 if pulse or gaze or "blink" in active else 0
        channels += 1 if "pitch_rise" in active or "voice_energy" in active else 0
        channels += 1 if "vrms_burst" in active or "motion_energy" in active else 0
        channels += 1 if ling_flags else 0
        disagree = channels >= 2 and score < 0.35
        if disagree and not (face_only or motion_only or voice_only):
            thoughts.append("Thought: more than one channel fired but fused load is low — modalities disagree.")

        review = False
        if not calibrated or (rppg.get("bpm") or 0) < 40:
            action = "need_more_data"
            thoughts.append("Action: need_more_data (pulse not stable).")
        elif face_only or motion_only or voice_only:
            action = "continue"
            thoughts.append("Action: continue; weak single-channel / face-only evidence — never review.")
        elif (z_true is not None and z_true >= 1.8 and (pulse or gaze)):
            action = "flag_review"
            review = True
            thoughts.append("Action: flag_review (large deviation from truth-talk with pulse/gaze). Not a lie verdict.")
        elif disagree or (0.40 <= score < 0.60 and channels >= 2):
            action = "flag_review"
            review = True
            thoughts.append("Action: flag_review for a human (borderline or discordant cues).")
        elif score >= 0.55 and (pulse or gaze):
            action = "continue"
            thoughts.append("Action: continue; elevated cue load with a supported channel. Not a lie verdict.")
        else:
            action = "continue"
            thoughts.append("Action: continue.")

        explanation = " ".join(thoughts[-3:])
        self.last = {
            "thoughts": thoughts,
            "action": action,
            "review": review,
            "explanation": explanation,
            "channels": channels,
        }
        return self.last
