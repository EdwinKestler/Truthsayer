# What this system can and cannot infer

Truthsayer is a **multimodal cue dashboard**. It estimates signals that the deception-detection literature has studied (heart rate, gaze, blinks, facial motion, voice pitch). It does **not** output a scientifically validated probability that someone is lying.

## Consensus from reviews

- Humans detect lies at about chance once a speaker crosses a “truth-default” threshold (Truth-Default Theory; see Convertino et al., 2024 review in *Acta Psychologica*).
- DePaulo et al. (2003) meta-analyzed 150+ proposed cues. Median effect size was about *d* = 0.10 — real but small.
- Expert survey work (Taylor & Francis, 2025) rates **cognitive-load** accounts as the most robust and **micro-expressions as the least**: 74.5% of experts said micro-expression theory has little or no scientific support.
- Luke (2019, *Perspectives on Psychological Science*) argues many published nonverbal “lie cues” are consistent with noise, small samples, and selective reporting.
- There is **no unique autonomic signature of deception**. Stress, cognitive load, fear of being disbelieved, and lying can look alike (NAS / polygraph reviews; MITRE 2005).

## Evidence for the cues we actually measure

| Cue in the app | What we compute | What the literature says |
|---|---|---|
| Visual facial BPM (rPPG / POS) | Plane-Orthogonal-to-Skin pulse from cheek RGB (Wang et al., IEEE TBME 2017) | On the DDPM interview set, POS heart-rate MAE was **3.16 bpm**. Pulse + simple rules reached ~63% question-level accuracy — better than micro-expressions, still far from forensic grade (Speth et al., 2021, arXiv:2106.06583). |
| Gaze / saccade-like shifts | Iris offset vs eye corners | Non-visual saccade *rate* rose in deceptive *answers* for 64% of DDPM subjects (p = 0.0098), matching Vrij et al. (2015). Discrimination by median split was only 54.5%. |
| Blink rate change | Eye aspect ratio over a 120-frame window | Mixed. Can reflect arousal or concentration. Used as a relative change vs the subject’s own baseline. |
| Lip compression / hand-on-face | Landmark aspect ratio; finger-in-face polygon | Self-adaptor / control behaviors. Weak, non-specific. |
| Motion magnification | Eulerian temporal bandpass on a face crop (Wu et al., SIGGRAPH 2012) | Amplifies micro-movements so a human (or a later classifier) can *see* them. Used in micro-expression *recognition* pipelines (Peng et al., 2019; Li et al.). **Spotting micro-expressions did not predict deceit on DDPM** (paired t-tests n.s.). |
| VRMS | RMS of MediaPipe landmark velocity | Compact facial-motion energy. Bursts can be speech, expressions, or micro-movements. Not a lie label. |
| Voice pitch (F0) | Autocorrelation 80–400 Hz | Pitch rises with anger/fear in ~70% of people (Ekman). Raised pitch is **not** a sign of deceit by itself — innocents also get aroused. |
| Voice RMS + peak frequency | Frame energy (dB) and rFFT peak | Used in speech-stress studies. Commercial “voice stress” / LVA systems performed at **chance** with high false positives (Harnsberger et al., 2009, *J. Forensic Sci.*). |
| FER mood | MTCNN + FER | Coarse displayed emotion. Easy to pose; not a veracity signal. |
| Cue-load fusion | Weighted OR of the above | DDPM found pulse and saccades were **not orthogonal**; fusion did not beat pulse alone. Our bar is an **arousal concordance** index, labeled as such. |
| Utterance window | Cue stats over a voiced statement (~0.6 s min, 0.8 s silence to close) | GLD (Rodriguez Diaz et al., 2021, arXiv:2104.12345) labels *statements*, not isolated frames. Their live rule used frame fraction; we keep the window, not their VGG lie classifier. |
| Static-face CNN “P(lie)” | **Not used** | On GLD, VGG+FC generalization is **57.4%** vs a **~58.4%** majority-class baseline (always “truth”). Cross-task accuracy falls to ~44%. A universal face-lie net is not a product feature. |

## Design rules we follow because of that evidence

1. **Baseline every subject with a talking calibration block**, not silent staring and not the whole session. Default: ~30 s factual general-knowledge answers (truth-talk) then ~30 s instructed false answers (directed-fabrication), then **freeze**. Interview utterances are z-scored against truth-talk and compared to the fabrication template. Silent 120-frame warmup remains only if `--cal-true 0 --cal-lie 0`.
2. **Prefer rPPG and gaze over micro-expression counts.** That matches DDPM’s own baselines.
3. **Never print “lying” / “deceit.”** The HUD says *cue load*. The old meter graphic is a research remnant; the marker now tracks cue load, not guilt.
4. **Voice is optional.** Microphone analysis is off until `--voice`, because it is the modality with the worst independent validation.
5. **Face/motion-only clusters never trigger human review.** GLD shows static-face deep nets do not transfer across people or tasks (arXiv:2104.12345v2).
6. **Score utterances, not isolated frames**, when speech (or a motion proxy) is present.
7. **This is not a polygraph substitute** and must not be used as the sole basis for hiring, immigration, or criminal decisions.

## Key references

- Speth, Vance, Czajka, Bowyer, Wright, Flynn (2021). *Deception Detection and Remote Physiological Monitoring.* arXiv:2106.06583.
- Wang, den Brinker, Stuijk, de Haan (2017). Algorithmic principles of remote PPG. *IEEE Trans. Biomedical Engineering.*
- de Haan & Jeanne (2013). Robust pulse rate from chrominance-based rPPG. *IEEE TBME.*
- Wu et al. (2012). Eulerian Video Magnification. *ACM SIGGRAPH / CACM.*
- DePaulo et al. (2003). Cues to deception. *Psychological Bulletin.*
- Vrij, Oliveira, Hammond, Ehrlichman (2015). Saccadic eye movement rate as a cue to deceit. *JARMAC.*
- Harnsberger, Hollien, Martin, Hollien (2009). Evaluating Layered Voice Analysis. *J. Forensic Sciences.*
- Ekman. Vocal clues to emotion / deceit (Paul Ekman Group overview).
- Bondi et al. (2023). Detecting deceptive behaviours through facial cues from videos: a systematic review. *Applied Sciences* 13:9188.
- Rodriguez Diaz, Aspandi, Sukno, Binefa (2021). Machine Learning-based Lie Detector applied to a Novel Annotated Game Dataset. arXiv:2104.12345v2. Statement-level GT from a competitive card game; static-face VGG generalization ≤ majority class; cross-domain collapse.
- Verschuere, Bogaard, et al. Comparable truth baselines (CTB) for within-subject comparison of later statements to a known truthful answer by the same person.
- Honts & Reavy (2015). Directed-lie vs probable-lie comparison questions in a mock-crime CQT; DLC is more standardized. **Not** used here as “relevant > control ⇒ deceptive.”
