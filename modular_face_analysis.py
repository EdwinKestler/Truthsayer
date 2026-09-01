import argparse
import cv2
import mediapipe as mp
from datetime import datetime
from matplotlib import pyplot as plt
import mss
import numpy as np
from scipy.signal import find_peaks
from scipy.spatial import distance as dist
from fer import FER
import threading
import time

# Constants
MAX_FRAMES = 120
RECENT_FRAMES = int(MAX_FRAMES / 10)
EYE_BLINK_HEIGHT = .15
SIGNIFICANT_BPM_CHANGE = 8
LIP_COMPRESSION_RATIO = .35
TELL_MAX_TTL = 30
TEXT_HEIGHT = 30
FACEMESH_FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]
EPOCH = time.time()

class GlobalState:
    def __init__(self):
        self.recording = None
        self.tells = {}
        self.blinks = [False] * MAX_FRAMES
        self.blinks2 = [False] * MAX_FRAMES  # for mirroring
        self.hand_on_face = [False] * MAX_FRAMES
        self.hand_on_face2 = [False] * MAX_FRAMES  # for mirroring
        self.face_area_size = 0
        self.hr_times = list(range(0, MAX_FRAMES))
        self.hr_values = [400] * MAX_FRAMES
        self.avg_bpms = [0] * MAX_FRAMES
        self.gaze_values = [0] * MAX_FRAMES
        self.calculating_mood = False
        self.mood = ''
        self.meter = cv2.imread('meter.png')
        self.fig = None
        self.ax = None
        self.line = None
        self.peakpts = None

class HeartRateMonitor:
    def __init__(self, global_state):
        self.global_state = global_state

    def chart_setup(self):
        plt.ion()
        self.global_state.fig = plt.figure()
        self.global_state.ax = self.global_state.fig.add_subplot(1, 1, 1)
        self.global_state.ax.set(ylim=(185, 200))
        self.global_state.line, = self.global_state.ax.plot(self.global_state.hr_times, self.global_state.hr_values, 'b-')
        self.global_state.peakpts, = self.global_state.ax.plot([], [], 'r+')

    def _bpm_display_from_buffer(self):
        valid = [bpm for bpm in self.global_state.avg_bpms if bpm > 0]
        if valid:
            return "BPM: {} ({})".format(int(valid[-1]), len(valid)), ""
        return "BPM: ...", ""

    def get_bpm_tells(self, cheekL, cheekR, fps, bpm_chart):
        gs = self.global_state
        if (cheekL is None or cheekR is None
                or cheekL.size == 0 or cheekR.size == 0
                or cheekL.ndim < 3 or cheekR.ndim < 3):
            return self._bpm_display_from_buffer()

        sample = np.average(cheekL[:, :, 1:3]) + np.average(cheekR[:, :, 1:3])
        if not np.isfinite(sample):
            return self._bpm_display_from_buffer()

        gs.hr_values = gs.hr_values[1:] + [float(sample)]

        if not fps:
            gs.hr_times = gs.hr_times[1:] + [time.time() - EPOCH]

        if bpm_chart:
            gs.line.set_data(gs.hr_times, gs.hr_values)
            gs.ax.relim()
            gs.ax.autoscale()

        if not np.all(np.isfinite(gs.hr_values)):
            return self._bpm_display_from_buffer()

        peaks, _ = find_peaks(gs.hr_values, threshold=.1, distance=5, prominence=.5, wlen=10)
        peak_times = [gs.hr_times[i] for i in peaks]

        if bpm_chart:
            gs.peakpts.set_data(peak_times, [gs.hr_values[i] for i in peaks])

        ibis = np.diff(peak_times)
        ibis = ibis[ibis > 0]
        if ibis.size == 0:
            bpms = np.array([])
        elif fps:
            bpms = 60.0 * float(fps) / ibis
        else:
            bpms = 60.0 / ibis
        bpms = bpms[(bpms > 50) & (bpms < 150)]
        recent_bpms = bpms[-3 * RECENT_FRAMES:]

        recent_avg_bpm = 0
        bpm_display = "BPM: ..."
        if recent_bpms.size > 1:
            recent_avg_bpm = int(np.average(recent_bpms))
            bpm_display = "BPM: {} ({})".format(recent_avg_bpm, len(recent_bpms))

        gs.avg_bpms = gs.avg_bpms[1:] + [recent_avg_bpm]

        bpm_change = ""
        if len(recent_bpms) > 2:
            all_bpms = [bpm for bpm in gs.avg_bpms if bpm > 0]
            if all_bpms:
                all_avg_bpm = sum(all_bpms) / len(all_bpms)
                avg_recent_bpm = sum(recent_bpms) / len(recent_bpms)
                bpm_delta = avg_recent_bpm - all_avg_bpm
                if bpm_delta > SIGNIFICANT_BPM_CHANGE:
                    bpm_change = "Heart rate increasing"
                elif bpm_delta < -SIGNIFICANT_BPM_CHANGE:
                    bpm_change = "Heart rate decreasing"

        return bpm_display, bpm_change

class GazeDetector:
    def __init__(self, global_state):
        self.global_state = global_state

    def get_avg_gaze(self, face):
        gaze_left = self.get_gaze(face, 476, 474, 263, 362)
        gaze_right = self.get_gaze(face, 471, 469, 33, 133)
        return round((gaze_left + gaze_right) / 2, 1)

    def get_gaze(self, face, iris_L_side, iris_R_side, eye_L_corner, eye_R_corner):
        iris = ((face[iris_L_side].x + face[iris_R_side].x) / 2,
                (face[iris_L_side].y + face[iris_R_side].y) / 2)
        eye_center = ((face[eye_L_corner].x + face[eye_R_corner].x) / 2,
                      (face[eye_L_corner].y + face[eye_R_corner].y) / 2)
        gaze_dist = dist.euclidean(iris, eye_center)
        eye_width = abs(face[eye_R_corner].x - face[eye_L_corner].x)
        if eye_width == 0:
            return 0
        gaze_relative = gaze_dist / eye_width

        if (eye_center[0] - iris[0]) < 0:  # flip along x for looking L vs R
            gaze_relative *= -1

        return gaze_relative

    def detect_gaze_change(self, avg_gaze):
        self.global_state.gaze_values = self.global_state.gaze_values[1:] + [avg_gaze]
        gaze_relative_matches = 1.0 * self.global_state.gaze_values.count(avg_gaze) / MAX_FRAMES
        if gaze_relative_matches < .01:  # looking in a new direction
            return gaze_relative_matches
        return 0

class MoodDetector:
    def __init__(self, global_state):
        self.global_state = global_state
        self.emotion_detector = FER(mtcnn=True)
        self._lock = threading.Lock()

    def maybe_start(self, image):
        with self._lock:
            if self.global_state.calculating_mood:
                return
            self.global_state.calculating_mood = True
        threading.Thread(target=self._detect, args=(image.copy(),), daemon=True).start()

    def _detect(self, image):
        try:
            detected_mood, score = self.emotion_detector.top_emotion(image)
            with self._lock:
                if score and (score > .4 or detected_mood == 'neutral'):
                    self.global_state.mood = detected_mood
        finally:
            with self._lock:
                self.global_state.calculating_mood = False

class BlinkDetector:
    def __init__(self, global_state):
        self.global_state = global_state

    @staticmethod
    def get_aspect_ratio(top, bottom, right, left):
        height = dist.euclidean([top.x, top.y], [bottom.x, bottom.y])
        width = dist.euclidean([right.x, right.y], [left.x, left.y])
        if width == 0:
            return 0
        return height / width

    def is_blinking(self, face):
        eyeR = [face[p] for p in [159, 145, 133, 33]]
        eyeL = [face[p] for p in [386, 374, 362, 263]]
        eyeR_ar = self.get_aspect_ratio(*eyeR)
        eyeL_ar = self.get_aspect_ratio(*eyeL)
        eyeA_ar = (eyeR_ar + eyeL_ar) / 2
        return eyeA_ar < EYE_BLINK_HEIGHT

    def get_blink_tell(self, blinks):
        if sum(blinks[:RECENT_FRAMES]) < 3:
            return None
        recent_closed = 1.0 * sum(blinks[-RECENT_FRAMES:]) / RECENT_FRAMES
        avg_closed = 1.0 * sum(blinks) / MAX_FRAMES
        if recent_closed > (20 * avg_closed):
            return "Increased blinking"
        elif avg_closed > (20 * recent_closed):
            return "Decreased blinking"
        else:
            return None

def decrement_tells(tells):
  for key, tell in tells.copy().items():
    if 'ttl' in tell:
      tell['ttl'] -= 1
      if tell['ttl'] <= 0:
        del tells[key]
  return tells


def main():
    global_state = GlobalState()
    global TELL_MAX_TTL

    hr_monitor = HeartRateMonitor(global_state)
    gaze_detector = GazeDetector(global_state)
    mood_detector = MoodDetector(global_state)
    blink_detector = BlinkDetector(global_state)

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', nargs='*', help='Input video device (number or path), file, or screen dimensions (x y width height), defaults to 0', default=['0'])
    parser.add_argument('--landmarks', '-l', help='Set to any value to draw face and hand landmarks')
    parser.add_argument('--bpm', '-b', help='Set to any value to draw color chart for heartbeats')
    parser.add_argument('--flip', '-f', help='Set to any value to flip resulting output (selfie view)')
    parser.add_argument('--ttl', '-t', help='How many frames for each displayed "tell" to last, defaults to 30', default='30')
    parser.add_argument('--record', '-r', help='Set to any value to save a timestamped AVI in current directory')
    parser.add_argument('--second', '-s', help='Secondary video input device (number or path)')
    args = parser.parse_args()

    if len(args.input) == 1:
        INPUT = int(args.input[0]) if args.input[0].isdigit() else args.input[0]
    elif len(args.input) != 4:
        return print("Wrong number of values for 'input' argument; should be 0, 1, or 4.")

    DRAW_LANDMARKS = args.landmarks is not None
    BPM_CHART = args.bpm is not None
    FLIP = args.flip is not None
    if args.ttl and args.ttl.isdigit():
        TELL_MAX_TTL = int(args.ttl)
    RECORD = args.record is not None

    cap2 = None
    if args.second is not None:
        second_src = int(args.second) if args.second.isdigit() else args.second
        cap2 = cv2.VideoCapture(second_src)

    if BPM_CHART:
        hr_monitor.chart_setup()

    calibrated = False
    calibration_frames = 0

    def handle_frame(image, fps=None):
        nonlocal calibration_frames, calibrated
        calibration_frames += process(
            image, face_mesh, hands, global_state, blink_detector,
            DRAW_LANDMARKS, BPM_CHART, FLIP, fps, hr_monitor, gaze_detector,
            mood_detector, calibrated)
        calibrated = (calibration_frames >= MAX_FRAMES)
        if cap2 is not None:
            process_second(cap2, image, face_mesh, hands, blink_detector, global_state)
        cv2.imshow('face', image)
        if global_state.recording is not None:
            global_state.recording.write(image)
        return cv2.waitKey(1) & 0xFF == ord('q')

    try:
        with mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh, mp.solutions.hands.Hands(max_num_hands=2, min_detection_confidence=0.7) as hands:
            if len(args.input) == 4:
                screen = {
                    "top": int(args.input[0]),
                    "left": int(args.input[1]),
                    "width": int(args.input[2]),
                    "height": int(args.input[3])
                }
                if RECORD:
                    RECORDING_FILENAME = str(datetime.now()).replace('.', '').replace(':', '') + '.avi'
                    global_state.recording = cv2.VideoWriter(
                        RECORDING_FILENAME, cv2.VideoWriter_fourcc(*'MJPG'), 10,
                        (screen["width"], screen["height"]))
                with mss.mss() as sct:
                    while True:
                        image = np.array(sct.grab(screen))[:, :, :3]  # drop alpha, remain BGR
                        if handle_frame(image):
                            break
            else:
                cap = cv2.VideoCapture(INPUT)
                fps = None
                if isinstance(INPUT, str) and INPUT.find('.') > -1:  # from file
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    print("FPS:", fps)
                else:  # from device
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                    cap.set(cv2.CAP_PROP_FPS, 30)

                if RECORD:
                    RECORDING_FILENAME = str(datetime.now()).replace('.', '').replace(':', '') + '.avi'
                    FRAME_SIZE = (int(cap.get(3)), int(cap.get(4)))
                    global_state.recording = cv2.VideoWriter(
                        RECORDING_FILENAME, cv2.VideoWriter_fourcc(*'MJPG'), 10, FRAME_SIZE)

                while cap.isOpened():
                    success, image = cap.read()
                    if not success:
                        break
                    if handle_frame(image, fps):
                        break
                cap.release()
    finally:
        if cap2 is not None:
            cap2.release()
        if global_state.recording is not None:
            global_state.recording.release()
        cv2.destroyAllWindows()


def new_tell(result):
  global TELL_MAX_TTL

  return {
    'text': result,
    'ttl': TELL_MAX_TTL
  }

def draw_on_frame(image, face_landmarks, hands_landmarks):
  mp.solutions.drawing_utils.draw_landmarks(
      image,
      face_landmarks,
      mp.solutions.face_mesh.FACEMESH_CONTOURS,
      landmark_drawing_spec=None,
      connection_drawing_spec=mp.solutions.drawing_styles
      .get_default_face_mesh_contours_style())
  mp.solutions.drawing_utils.draw_landmarks(
      image,
      face_landmarks,
      mp.solutions.face_mesh.FACEMESH_IRISES,
      landmark_drawing_spec=None,
      connection_drawing_spec=mp.solutions.drawing_styles
      .get_default_face_mesh_iris_connections_style())
  for hand_landmarks in (hands_landmarks or []):
    mp.solutions.drawing_utils.draw_landmarks(
        image,
        hand_landmarks,
        mp.solutions.hands.HAND_CONNECTIONS,
        mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
        mp.solutions.drawing_styles.get_default_hand_connections_style())

def add_text(image, tells, calibrated, global_state):
    text_y = TEXT_HEIGHT
    if global_state.mood:
        write("Mood: {}".format(global_state.mood), image, int(.75 * image.shape[1]), TEXT_HEIGHT)
    if calibrated:
        for tell in tells.values():
            write(tell['text'], image, 10, text_y)
            text_y += TEXT_HEIGHT

def write(text, image, x, y):
    cv2.putText(img=image, text=text, org=(x, y),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=[0, 0, 0],
                lineType=cv2.LINE_AA, thickness=4)
    cv2.putText(img=image, text=text, org=(x, y),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=[255, 255, 255],
                lineType=cv2.LINE_AA, thickness=2)

def get_area(image, draw, topL, topR, bottomR, bottomL):
    h, w = image.shape[:2]
    topY = int((topR.y+topL.y)/2 * h)
    botY = int((bottomR.y+bottomL.y)/2 * h)
    leftX = int((topL.x+bottomL.x)/2 * w)
    rightX = int((topR.x+bottomR.x)/2 * w)
    if draw:
        cv2.circle(image, (leftX,topY), 2, (255,0,0), 2)
        cv2.circle(image, (leftX,botY), 2, (255,0,0), 2)
        cv2.circle(image, (rightX,topY), 2, (255,0,0), 2)
        cv2.circle(image, (rightX,botY), 2, (255,0,0), 2)

    x1, x2 = sorted((leftX, rightX))
    y1, y2 = sorted((topY, botY))
    x1 = max(0, min(x1, w))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h))
    y2 = max(0, min(y2, h))
    if (x2 - x1) < 2 or (y2 - y1) < 2:
        return None
    return image[y1:y2, x1:x2]

def check_hand_on_face(hands_landmarks, face):
  if hands_landmarks:
    face_landmarks = [face[p] for p in FACEMESH_FACE_OVAL]
    face_points = [[[p.x, p.y] for p in face_landmarks]]
    face_contours = np.array(face_points).astype(np.single)

    for hand_landmarks in hands_landmarks:
      hand = []
      for point in hand_landmarks.landmark:
        hand.append( (point.x, point.y) )

      for finger in [4, 8, 20]:
        overlap = cv2.pointPolygonTest(face_contours, hand[finger], False)
        if overlap != -1:
          return True
  return False

def get_lip_ratio(face, blink_detector):
    return blink_detector.get_aspect_ratio(face[0], face[17], face[61], face[291])

def add_truth_meter(image, tell_count, global_state):
    if global_state.meter is None:
        return
    width = image.shape[1]
    sm = int(width / 64)
    bg = int(width / 3.2)

    resized_meter = cv2.resize(global_state.meter, (bg, sm), interpolation=cv2.INTER_AREA)
    image[sm:(sm+sm), bg:(bg+bg), 0:3] = resized_meter[:, :, 0:3]

    if tell_count:
        tellX = bg + int(bg / 4) * (tell_count - 1)
        cv2.rectangle(image, (tellX, int(.9 * sm)), (tellX + int(sm / 2), int(2.1 * sm)), (0, 0, 0), 2)

def get_face_relative_area(face):
  face_width = abs(max(face[454].x, 0) - max(face[234].x, 0))
  face_height = abs(max(face[152].y, 0) - max(face[10].y, 0))
  return face_width * face_height

def find_face_and_hands(image_original, face_mesh, hands):
  image = image_original.copy()
  image.flags.writeable = False # pass by reference to improve speed
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  faces = face_mesh.process(image)
  hands_landmarks = hands.process(image).multi_hand_landmarks

  face_landmarks = None
  if faces.multi_face_landmarks and len(faces.multi_face_landmarks) > 0:
    face_landmarks = faces.multi_face_landmarks[0] # use first face found

  return face_landmarks, hands_landmarks

def process(image, face_mesh, hands, global_state, blink_detector, draw, bpm_chart, flip, fps, hr_monitor, gaze_detector, mood_detector, calibrated):
    global_state.tells = decrement_tells(global_state.tells)

    face_landmarks, hands_landmarks = find_face_and_hands(image, face_mesh, hands)
    if face_landmarks:
        face = face_landmarks.landmark
        global_state.face_area_size = get_face_relative_area(face)
        mood_detector.maybe_start(image)

        cheekL = get_area(image, draw, topL=face[449], topR=face[350], bottomR=face[429], bottomL=face[280])
        cheekR = get_area(image, draw, topL=face[121], topR=face[229], bottomR=face[50], bottomL=face[209])

        bpm_display, bpm_change = hr_monitor.get_bpm_tells(cheekL, cheekR, fps, bpm_chart)
        global_state.tells['avg_bpms'] = new_tell(bpm_display)
        if bpm_change:
            global_state.tells['bpm_change'] = new_tell(bpm_change)

        global_state.blinks = global_state.blinks[1:] + [blink_detector.is_blinking(face)]
        recent_blink_tell = blink_detector.get_blink_tell(global_state.blinks)
        if recent_blink_tell:
            global_state.tells['blinking'] = new_tell(recent_blink_tell)

        recent_hand_on_face = check_hand_on_face(hands_landmarks, face)
        global_state.hand_on_face = global_state.hand_on_face[1:] + [recent_hand_on_face]
        if recent_hand_on_face:
            global_state.tells['hand'] = new_tell("Hand covering face")

        avg_gaze = gaze_detector.get_avg_gaze(face)
        if gaze_detector.detect_gaze_change(avg_gaze):
            global_state.tells['gaze'] = new_tell("Change in gaze")

        lip_compression = get_lip_ratio(face, blink_detector)
        if lip_compression < LIP_COMPRESSION_RATIO:
            global_state.tells['lips'] = new_tell("Lip compression")

        if bpm_chart:
            global_state.fig.canvas.draw()
            global_state.fig.canvas.flush_events()

        if draw:
            draw_on_frame(image, face_landmarks, hands_landmarks)

    if flip:
        cv2.flip(image, 1, dst=image)

    add_text(image, global_state.tells, calibrated, global_state)
    add_truth_meter(image, len(global_state.tells), global_state)

    return 1 if (face_landmarks and not calibrated) else 0

def mirror_compare(first, second, rate, less, more):
  if (rate * first) < second:
    return less
  elif first > (rate * second):
    return more
  return None

def get_blink_comparison(blinks1, blinks2):
  return mirror_compare(sum(blinks1), sum(blinks2), 1.8, "Blink less", "Blink more")

def get_hand_face_comparison(hand1, hand2):
  return mirror_compare(sum(hand1), sum(hand2), 2.1, "Stop touching face", "Touch face more")

def get_face_size_comparison(ratio1, ratio2):
  return mirror_compare(ratio1, ratio2, 1.5, "Too close", "Too far")


def process_second(cap, image, face_mesh, hands, blink_detector, global_state):
    success2, image2 = cap.read()
    if success2:
        face_landmarks2, hands_landmarks2 = find_face_and_hands(image2, face_mesh, hands)

        if face_landmarks2:
            face2 = face_landmarks2.landmark

            global_state.blinks2 = global_state.blinks2[1:] + [blink_detector.is_blinking(face2)]
            blink_mirror = get_blink_comparison(global_state.blinks, global_state.blinks2)

            global_state.hand_on_face2 = global_state.hand_on_face2[1:] + [check_hand_on_face(hands_landmarks2, face2)]
            hand_face_mirror = get_hand_face_comparison(global_state.hand_on_face, global_state.hand_on_face2)

            face_area_size2 = get_face_relative_area(face2)
            face_ratio_mirror = get_face_size_comparison(global_state.face_area_size, face_area_size2)

            text_y = 2 * TEXT_HEIGHT  # show prompts below 'mood' on right side
            for comparison in [blink_mirror, hand_face_mirror, face_ratio_mirror]:
                if comparison:
                    write(comparison, image, int(.75 * image.shape[1]), text_y)
                    text_y += TEXT_HEIGHT


if __name__ == '__main__':
    main()
