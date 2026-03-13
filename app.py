from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import json
import os
from datetime import datetime
from collections import deque
import uuid

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
UPLOAD_FOLDER = '/tmp/dayowl_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

all_sessions = []

class DrawerDetector:
    def __init__(self, sensitivity=20):
        self.sensitivity = sensitivity
        self.baseline = None
        self.cal_frames = []
        self.calibrated = False
        self.drawer_open = False
        self.open_time = None
        self.events = []

    def process(self, frame, timestamp):
        h, w = frame.shape[:2]
        region = frame[int(h*0.5):h, int(w*0.2):int(w*0.8)]
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY).astype(float)

        if not self.calibrated:
            self.cal_frames.append(gray)
            if len(self.cal_frames) >= 15:
                self.baseline = np.mean(self.cal_frames, axis=0)
                self.calibrated = True
            return False, False, False, 0.0

        diff = np.mean(np.abs(gray - self.baseline))
        is_open = diff > self.sensitivity
        just_opened = is_open and not self.drawer_open
        just_closed = not is_open and self.drawer_open

        if just_opened:
            self.open_time = timestamp
            self.events.append({'event': 'opened', 'time': round(timestamp,1), 'diff_score': round(diff,1)})
        if just_closed and self.open_time:
            self.events.append({'event': 'closed', 'time': round(timestamp,1),
                                'duration_sec': round(timestamp - self.open_time, 1), 'diff_score': round(diff,1)})

        self.drawer_open = is_open
        return is_open, just_opened, just_closed, round(diff, 1)


class MotionDetector:
    def __init__(self, sensitivity=40):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=sensitivity)

    def detect(self, frame):
        fg_mask = self.bg_subtractor.apply(frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        return round((np.sum(fg_mask > 0) / (frame.shape[0] * frame.shape[1])) * 100, 2)


def analyze_video(video_path, settings):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {'error': 'Cannot open video file'}

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration     = total_frames / fps
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    drawer_sens    = int(settings.get('drawer_sensitivity', 20))
    motion_sens    = int(settings.get('motion_sensitivity', 40))
    no_sale_window = int(settings.get('no_sale_window_sec', 120))
    drawer_timeout = int(settings.get('drawer_timeout_sec', 60))
    what_happened  = settings.get('what_happened', '').strip()

    # Parse manual transaction times
    txn_times = []
    raw_txn = settings.get('txn_times', '').strip()
    if raw_txn:
        try:
            txn_times = [float(t.strip()) for t in raw_txn.split(',') if t.strip()]
        except:
            txn_times = []
    auto_txn = not bool(txn_times)
    if auto_txn:
        txn_times = [round(duration * 0.3, 1), round(duration * 0.6, 1)]

    drawer_det = DrawerDetector(sensitivity=drawer_sens)
    motion_det = MotionDetector(sensitivity=motion_sens)

    last_txn_time = None
    txn_count = 0
    session_alerts = []
    frame_log = []
    drawer_timeout_alerted = False

    frame_num = 0
    skip = max(1, int(fps / 5))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1
        if frame_num % skip != 0:
            continue

        timestamp = round(frame_num / fps, 2)

        for t in txn_times:
            if abs(timestamp - t) < (skip / fps) * 1.5:
                last_txn_time = timestamp
                txn_count += 1

        secs_since_txn = round(timestamp - last_txn_time, 1) if last_txn_time else 9999
        is_open, just_opened, just_closed, diff_score = drawer_det.process(frame, timestamp)
        motion_pct = motion_det.detect(frame)

        # ALERT: Drawer open — no sale
        if just_opened and secs_since_txn > no_sale_window:
            session_alerts.append({
                'id': str(uuid.uuid4())[:8],
                'type': 'DRAWER OPEN — NO SALE',
                'severity': 'CRITICAL',
                'message': f'Drawer opened {int(secs_since_txn)}s after last transaction. Your threshold: {no_sale_window}s.',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'video_time': f'{timestamp}s',
                'diff_score': diff_score,
                'why': f'diff_score {diff_score} > drawer_sensitivity {drawer_sens}'
            })

        # ALERT: Drawer open too long
        if is_open and drawer_det.open_time and not drawer_timeout_alerted:
            open_dur = timestamp - drawer_det.open_time
            if open_dur > drawer_timeout:
                session_alerts.append({
                    'id': str(uuid.uuid4())[:8],
                    'type': 'DRAWER OPEN TOO LONG',
                    'severity': 'WARNING',
                    'message': f'Drawer open for {int(open_dur)}s. Your threshold: {drawer_timeout}s.',
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'video_time': f'{timestamp}s',
                    'diff_score': diff_score,
                    'why': f'open duration {int(open_dur)}s > drawer_timeout {drawer_timeout}s'
                })
                drawer_timeout_alerted = True

        if just_closed:
            drawer_timeout_alerted = False

        # Per-second log
        if frame_num % int(fps) == 0:
            frame_log.append({
                'time': f'{timestamp}s',
                'drawer': 'OPEN 🔴' if is_open else 'CLOSED 🟢',
                'diff_score': diff_score,
                'diff_vs_threshold': f'{diff_score} vs {drawer_sens} → {"OPEN" if is_open else "closed"}',
                'motion_pct': f'{motion_pct}%',
                'secs_since_txn': secs_since_txn if secs_since_txn != 9999 else 'no txn yet',
                'alert_fired': '🚨 YES' if any(a['video_time'] == f'{timestamp}s' for a in session_alerts) else 'no'
            })

    cap.release()

    result = {
        'session_id': str(uuid.uuid4())[:8],
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'video': os.path.basename(video_path),
        'video_info': {
            'duration_sec': round(duration, 1),
            'fps': round(fps, 1),
            'resolution': f'{width}x{height}',
            'frames_analyzed': len(frame_log)
        },
        'settings_used': {
            'drawer_sensitivity': drawer_sens,
            'motion_sensitivity': motion_sens,
            'no_sale_window_sec': no_sale_window,
            'drawer_timeout_sec': drawer_timeout,
            'transaction_times': txn_times,
            'auto_transactions': auto_txn
        },
        'what_happened': what_happened,
        'summary': {
            'total_alerts': len(session_alerts),
            'critical': sum(1 for a in session_alerts if a['severity'] == 'CRITICAL'),
            'warnings':  sum(1 for a in session_alerts if a['severity'] == 'WARNING'),
            'transactions_recorded': txn_count,
            'drawer_events': len(drawer_det.events),
        },
        'alerts': session_alerts,
        'drawer_events': drawer_det.events,
        'frame_log': frame_log
    }

    all_sessions.insert(0, result)
    if len(all_sessions) > 20:
        all_sessions.pop()

    return result


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400
    file = request.files['video']
    if not file.filename:
        return jsonify({'error': 'No file selected'}), 400

    filename = f'{uuid.uuid4().hex[:8]}_{file.filename}'
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    settings = {
        'drawer_sensitivity': request.form.get('drawer_sensitivity', 20),
        'motion_sensitivity': request.form.get('motion_sensitivity', 40),
        'no_sale_window_sec': request.form.get('no_sale_window_sec', 120),
        'drawer_timeout_sec': request.form.get('drawer_timeout_sec', 60),
        'what_happened':      request.form.get('what_happened', ''),
        'txn_times':          request.form.get('txn_times', ''),
    }

    result = analyze_video(filepath, settings)
    try:
        os.remove(filepath)
    except:
        pass
    return jsonify(result)


@app.route('/sessions')
def get_sessions():
    return jsonify([{
        'session_id':   s['session_id'],
        'timestamp':    s['timestamp'],
        'video':        s['video'],
        'total_alerts': s['summary']['total_alerts'],
        'critical':     s['summary']['critical'],
        'what_happened': s['what_happened']
    } for s in all_sessions])


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'sessions': len(all_sessions)})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
