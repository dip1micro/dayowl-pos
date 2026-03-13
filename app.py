from flask import Flask, render_template, request, jsonify, send_file
import cv2
import numpy as np
import base64
import os
from datetime import datetime
import uuid
import requests  # for Supabase REST API calls

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
UPLOAD_FOLDER = '/tmp/dayowl_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

all_sessions = []

# ── Supabase Config ────────────────────────────────────────────────────────────
SUPABASE_URL     = os.environ.get('SUPABASE_URL', 'https://ywiuvqtjvlajzgailajc.supabase.co')
SUPABASE_KEY     = os.environ.get('SUPABASE_SERVICE_KEY', '')  # set in Railway env vars
STORE_ID         = os.environ.get('STORE_ID', '85485ddc-d786-40a0-9f1b-5009513c1b6a')

def upload_snapshot_to_supabase(snapshot_b64, alert_id):
    """
    Upload a base64 snapshot image to Supabase Storage.
    Returns the public URL or None if failed.
    """
    if not SUPABASE_KEY or not snapshot_b64:
        return None
    try:
        img_bytes = base64.b64decode(snapshot_b64)
        filename  = f'{alert_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg'
        headers   = {
            'apikey':        SUPABASE_KEY,
            'Authorization': f'Bearer {SUPABASE_KEY}',
            'Content-Type':  'image/jpeg',
            'x-upsert':      'true',
        }
        resp = requests.post(
            f'{SUPABASE_URL}/storage/v1/object/snapshots/{filename}',
            data=img_bytes,
            headers=headers,
            timeout=10
        )
        if resp.status_code in (200, 201):
            public_url = f'{SUPABASE_URL}/storage/v1/object/public/snapshots/{filename}'
            print(f'[Supabase] Snapshot uploaded: {public_url}')
            return public_url
        else:
            print(f'[Supabase] Snapshot upload failed: {resp.status_code} {resp.text}')
            return None
    except Exception as e:
        print(f'[Supabase] Snapshot upload error: {e}')
        return None


def push_alert_to_supabase(alert):
    """
    Push a detected alert into the Supabase alerts table.
    Uploads snapshot image to Storage and saves public URL.
    """
    if not SUPABASE_KEY:
        print('[Supabase] SUPABASE_SERVICE_KEY not set — skipping push')
        return False

    # Upload snapshot image first
    snapshot_url = upload_snapshot_to_supabase(
        alert.get('snapshot_b64'), alert['id']
    )

    severity_map = {'CRITICAL': 'high', 'WARNING': 'medium'}

    payload = {
        'store_id':     STORE_ID,
        'severity':     severity_map.get(alert['severity'], 'medium'),
        'type':         alert['type'].lower().replace(' ', '_').replace('—', '').replace('-', '_'),
        'message':      alert['message'],
        'register':     'Video Analyzer',
        'cashier_name': None,
        'is_resolved':  False,
        'snapshot_url': snapshot_url,
    }

    headers = {
        'apikey':        SUPABASE_KEY,
        'Authorization': f'Bearer {SUPABASE_KEY}',
        'Content-Type':  'application/json',
        'Prefer':        'return=minimal',
    }

    try:
        resp = requests.post(
            f'{SUPABASE_URL}/rest/v1/alerts',
            json=payload,
            headers=headers,
            timeout=5
        )
        if resp.status_code in (200, 201):
            print(f'[Supabase] Alert pushed: {alert["type"]} | snapshot: {snapshot_url}')
            return True
        else:
            print(f'[Supabase] Push failed: {resp.status_code} {resp.text}')
            return False
    except Exception as e:
        print(f'[Supabase] Error pushing alert: {e}')
        return False


def push_transaction_to_supabase(tx_number, amount, tx_type, status, cashier=None, register=None):
    """
    Push a transaction record into the Supabase transactions table.
    """
    if not SUPABASE_KEY:
        return False

    payload = {
        'store_id':     STORE_ID,
        'tx_number':    tx_number,
        'amount':       amount,
        'type':         tx_type,
        'status':       status,
        'cashier_name': cashier,
        'register':     register,
    }

    headers = {
        'apikey':        SUPABASE_KEY,
        'Authorization': f'Bearer {SUPABASE_KEY}',
        'Content-Type':  'application/json',
        'Prefer':        'return=minimal',
    }

    try:
        resp = requests.post(
            f'{SUPABASE_URL}/rest/v1/transactions',
            json=payload,
            headers=headers,
            timeout=5
        )
        return resp.status_code in (200, 201)
    except Exception as e:
        print(f'[Supabase] Error pushing transaction: {e}')
        return False


# ── Helpers ────────────────────────────────────────────────────────────────────
def frame_to_base64(frame):
    """Convert OpenCV frame to base64 string for embedding in JSON"""
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode('utf-8')

def annotate_frame(frame, label, severity='CRITICAL'):
    """Draw alert label on frame"""
    annotated = frame.copy()
    h, w = annotated.shape[:2]
    color = (0, 0, 255) if severity == 'CRITICAL' else (0, 165, 255)
    cv2.rectangle(annotated, (0, 0), (w, 48), (0, 0, 0), -1)
    cv2.putText(annotated, f'DAYOWL ALERT: {label}',
                (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cv2.putText(annotated, ts, (w - 220, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    return annotated

def make_side_by_side(before_frame, after_frame, label):
    """Create before/after comparison image"""
    h = max(before_frame.shape[0], after_frame.shape[0])
    w = before_frame.shape[1]
    b = cv2.resize(before_frame, (w, h))
    a = cv2.resize(after_frame,  (w, h))
    cv2.rectangle(b, (0, 0), (w, 36), (30, 30, 30), -1)
    cv2.putText(b, 'BEFORE (drawer closed)', (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 1)
    cv2.rectangle(a, (0, 0), (w, 36), (30, 30, 30), -1)
    cv2.putText(a, f'ALERT: {label}', (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    divider = np.zeros((h, 4, 3), dtype=np.uint8)
    divider[:] = (80, 80, 80)
    combined = np.hstack([b, divider, a])
    return combined

def draw_detection_box(frame, sensitivity):
    """Draw the region the model is actually watching"""
    annotated = frame.copy()
    h, w = annotated.shape[:2]
    x1, y1 = int(w*0.2), int(h*0.5)
    x2, y2 = int(w*0.8), h
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(annotated, 'DETECTION ZONE', (x1+4, y1+20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    cv2.putText(annotated, f'sensitivity={sensitivity}', (x1+4, y1+42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    return annotated


# ── Detectors ─────────────────────────────────────────────────────────────────
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


# ── Main Analysis ──────────────────────────────────────────────────────────────
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
    prev_frame = None
    frame_buffer = []

    frame_num = 0
    skip = max(1, int(fps / 5))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        frame_buffer.append(frame.copy())
        if len(frame_buffer) > 25:
            frame_buffer.pop(0)

        if frame_num % skip != 0:
            continue

        timestamp = round(frame_num / fps, 2)

        for t in txn_times:
            if abs(timestamp - t) < (skip / fps) * 1.5:
                last_txn_time = timestamp
                txn_count += 1
                # ── Push transaction to Supabase ──────────────────────────
                push_transaction_to_supabase(
                    tx_number=f'VID-{str(uuid.uuid4())[:6].upper()}',
                    amount=0.00,
                    tx_type='no_sale',
                    status='review',
                    cashier=None,
                    register='Video Analyzer'
                )

        secs_since_txn = round(timestamp - last_txn_time, 1) if last_txn_time else 9999
        is_open, just_opened, just_closed, diff_score = drawer_det.process(frame, timestamp)
        motion_pct = motion_det.detect(frame)

        # ── ALERT: Drawer open — no sale ──────────────────────────────────
        if just_opened and secs_since_txn > no_sale_window:
            before_frame = frame_buffer[0] if frame_buffer else frame

            snapshot = annotate_frame(frame.copy(), 'DRAWER OPEN — NO SALE', 'CRITICAL')
            snapshot_b64 = frame_to_base64(snapshot)
            sbs = make_side_by_side(before_frame, frame.copy(), 'DRAWER OPEN')
            sbs_b64 = frame_to_base64(sbs)
            boxed = draw_detection_box(frame.copy(), drawer_sens)
            boxed_b64 = frame_to_base64(boxed)

            alert = {
                'id': str(uuid.uuid4())[:8],
                'type': 'DRAWER OPEN — NO SALE',
                'severity': 'CRITICAL',
                'message': f'Drawer opened {int(secs_since_txn)}s after last transaction. Threshold: {no_sale_window}s.',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'video_time': f'{timestamp}s',
                'diff_score': diff_score,
                'why': f'diff_score {diff_score} > sensitivity {drawer_sens}',
                'snapshot_b64': snapshot_b64,
                'sidebyside_b64': sbs_b64,
                'boxed_b64': boxed_b64,
            }
            session_alerts.append(alert)

            # ── Push to Supabase → appears live in DayOwl dashboard ───────
            push_alert_to_supabase(alert)

        # ── ALERT: Drawer open too long ───────────────────────────────────
        if is_open and drawer_det.open_time and not drawer_timeout_alerted:
            open_dur = timestamp - drawer_det.open_time
            if open_dur > drawer_timeout:
                snapshot = annotate_frame(frame.copy(), f'DRAWER OPEN {int(open_dur)}s', 'WARNING')
                snapshot_b64 = frame_to_base64(snapshot)
                boxed = draw_detection_box(frame.copy(), drawer_sens)
                boxed_b64 = frame_to_base64(boxed)

                alert = {
                    'id': str(uuid.uuid4())[:8],
                    'type': 'DRAWER OPEN TOO LONG',
                    'severity': 'WARNING',
                    'message': f'Drawer open for {int(open_dur)}s. Your threshold: {drawer_timeout}s.',
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'video_time': f'{timestamp}s',
                    'diff_score': diff_score,
                    'why': f'open duration {int(open_dur)}s > timeout {drawer_timeout}s',
                    'snapshot_b64': snapshot_b64,
                    'sidebyside_b64': None,
                    'boxed_b64': boxed_b64,
                }
                session_alerts.append(alert)

                # ── Push to Supabase ──────────────────────────────────────
                push_alert_to_supabase(alert)
                drawer_timeout_alerted = True

        if just_closed:
            drawer_timeout_alerted = False

        if frame_num % int(fps) == 0:
            thumb = cv2.resize(frame, (120, 68))
            thumb_b64 = frame_to_base64(thumb)
            frame_log.append({
                'time': f'{timestamp}s',
                'drawer': 'OPEN 🔴' if is_open else 'CLOSED 🟢',
                'diff_score': diff_score,
                'diff_vs_threshold': f'{diff_score} vs {drawer_sens}',
                'motion_pct': f'{motion_pct}%',
                'secs_since_txn': secs_since_txn if secs_since_txn != 9999 else 'no txn yet',
                'alert_fired': '🚨 YES' if any(a['video_time'] == f'{timestamp}s' for a in session_alerts) else 'no',
                'thumb_b64': thumb_b64,
            })

        prev_frame = frame.copy()

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


# ── Routes ─────────────────────────────────────────────────────────────────────
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
        'session_id':    s['session_id'],
        'timestamp':     s['timestamp'],
        'video':         s['video'],
        'total_alerts':  s['summary']['total_alerts'],
        'critical':      s['summary']['critical'],
        'what_happened': s['what_happened']
    } for s in all_sessions])

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'sessions': len(all_sessions)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
