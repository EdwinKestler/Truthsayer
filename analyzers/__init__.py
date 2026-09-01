"""Multimodal cue analyzers for Truthsayer.

These modules estimate *stress and arousal correlates*, not truth.
See docs/SCIENCE.md for evidence and limits.
"""

from .rppg import FacialRPPG
from .motion import MotionMagnifier
from .vrms import FacialVRMS
from .voice import VoiceAnalyzer
from .fusion import CueFusion
from .overlay import draw_hud
from .linguistics import LinguisticCues
from .session_log import SessionLog
from .utterance import UtteranceTracker, is_speaking
from .baseline import DualBaseline

__all__ = [
    "FacialRPPG",
    "MotionMagnifier",
    "FacialVRMS",
    "VoiceAnalyzer",
    "CueFusion",
    "draw_hud",
    "LinguisticCues",
    "SessionLog",
    "UtteranceTracker",
    "is_speaking",
    "DualBaseline",
]
