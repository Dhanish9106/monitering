# PersonMonitor — Technical Documentation

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Database Schema](#2-database-schema)
3. [Application Modules](#3-application-modules)
4. [API Routes](#4-api-routes)
5. [Authentication & Authorization](#5-authentication--authorization)
6. [Detection Engine](#6-detection-engine)
7. [Real-time Streaming](#7-real-time-streaming)
8. [Excel Export](#8-excel-export)
9. [AI Assistant](#9-ai-assistant)
10. [Frontend Pages](#10-frontend-pages)
11. [Configuration Reference](#11-configuration-reference)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Architecture Overview

```
Phone IP Camera (MJPEG stream)
        │
        ▼
  camera_worker() thread  ──►  YOLOv8m detection
        │                            │
        │                     annotated frame
        │                            │
        ▼                            ▼
  frame_lock (thread-safe)     SQLite detections table
        │                            │
        ▼                            ▼
  /video/<cam_id>            /events (SSE stream)
  MJPEG stream                       │
        │                            ▼
        └──────────────►  Browser Dashboard (real-time)
                                     │
                                     ▼
                            /ai (Gemini 2.5 Flash)
                         Role-scoped DB context
```

- Each camera runs in its own background daemon thread
- Frames are stored in a shared dictionary protected by `threading.Lock`
- The Flask main thread serves HTTP requests concurrently using `threaded=True`
- Real-time updates are pushed to the browser via Server-Sent Events (SSE) — no polling

---

## 2. Database Schema

Database file: `monitor.db` (SQLite, auto-created on first run)

### Table: `users`

| Column   | Type    | Description                        |
|----------|---------|------------------------------------|
| id       | INTEGER | Primary key, auto-increment        |
| username | TEXT    | Unique username                    |
| password | TEXT    | SHA-256 hashed password            |
| role     | TEXT    | `admin` or `user`                  |

### Table: `cameras`

| Column | Type    | Description                              |
|--------|---------|------------------------------------------|
| id     | INTEGER | Primary key, auto-increment              |
| name   | TEXT    | Display name (e.g. "Front Door")         |
| url    | TEXT    | Full stream URL (e.g. `http://ip:8080/video`) |
| active | INTEGER | `1` = active, `0` = soft-deleted         |

### Table: `detections`

| Column    | Type    | Description                          |
|-----------|---------|--------------------------------------|
| id        | INTEGER | Primary key, auto-increment          |
| cam_id    | INTEGER | Foreign key → cameras.id             |
| cam_name  | TEXT    | Camera display name (denormalized)   |
| count     | INTEGER | Number of persons detected           |
| timestamp | TEXT    | Detection time (`YYYY-MM-DD HH:MM:SS`) |

> Only records where `count > 0` are inserted and shown in the detection log.

---

## 3. Application Modules

### `app.py`

#### Global State

```python
cameras = {}
# cam_id -> {
#   "url":    str,               # stream URL
#   "name":   str,               # display name
#   "frame":  np.ndarray | None, # latest annotated frame
#   "count":  int,               # current person count
#   "time":   str,               # last detection timestamp
#   "lock":   threading.Lock,    # frame access lock
#   "active": bool               # thread control flag
# }
```

#### Key Functions

| Function | Description |
|---|---|
| `init_db()` | Creates all tables and default admin user on startup |
| `get_db()` | Returns a SQLite connection with `Row` factory |
| `hash_pw(pw)` | Returns SHA-256 hex digest of a password string |
| `camera_worker(cam_id, url, name)` | Background thread — reads frames, runs YOLO, writes to DB |
| `start_camera(cam_id, url, name)` | Initializes camera state and spawns worker thread |
| `stop_camera(cam_id)` | Sets `active=False` to stop the worker thread and removes from dict |
| `load_cameras_from_db()` | Called on startup — restarts all active cameras from DB |
| `logged_in()` | Returns `True` if session has a user |
| `is_admin()` | Returns `True` if session role is `admin` |
| `login_required(f)` | Decorator — redirects to login if not authenticated |
| `admin_required(f)` | Decorator — redirects to login if not admin |

---

## 4. API Routes

### Public

| Method | Route | Description |
|--------|-------|-------------|
| GET/POST | `/` | Login page |
| GET | `/logout` | Clears session, redirects to login |

### Authenticated (User + Admin)

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/dashboard` | Main monitoring dashboard |
| GET | `/video/<cam_id>` | MJPEG video stream for a camera |
| GET | `/events` | SSE stream — pushes live detection updates |
| GET | `/api/stats` | JSON — live counts + recent detections |
| GET | `/export` | Downloads detection history as `.xlsx` |
| POST | `/ai` | Sends query to Gemini with role-scoped DB context, returns AI answer |

### Admin Only

| Method | Route | Description |
|--------|-------|-------------|
| GET/POST | `/admin` | Admin panel — manage cameras and users |

#### POST Actions on `/admin`

| `action` value | Form Fields | Effect |
|---|---|---|
| `add_cam` | `name`, `ip` | Builds URL as `http://<ip>:8080/video`, saves to DB, starts camera thread |
| `delete_cam` | `cam_id` | Stops thread, sets `active=0` in DB |
| `add_user` | `uname`, `pw`, `role` | Creates new user with hashed password |
| `delete_user` | `uid` | Deletes user (cannot delete `admin`) |

#### SSE Event Format (`/events`)

```
data: <cam_id>|<count>|<timestamp>|<cam_name>
```

Example:
```
data: 3|2|2024-12-15 14:32:01|Front Door
```

#### `/api/stats` Response

```json
{
  "total": 142,
  "live": {
    "1": { "count": 2, "time": "2024-12-15 14:32:01", "name": "Front Door" }
  },
  "recent": [
    { "cam_name": "Front Door", "count": 2, "timestamp": "2024-12-15 14:32:01" }
  ]
}
```

---

## 5. Authentication & Authorization

- Passwords are hashed with **SHA-256** before storage — never stored in plain text
- Sessions use Flask's signed cookie session (`secret_key`)
- Login page has a **role toggle** (User / Admin) — the selected role is sent as `expected_role`
- If credentials are valid but the role doesn't match the selected tab, login is rejected with a specific error message
- `login_required` decorator protects all user-facing routes
- `admin_required` decorator protects admin-only routes

### Session Keys

| Key | Value |
|-----|-------|
| `session["user"]` | Username string |
| `session["role"]` | `"admin"` or `"user"` |

---

## 6. Detection Engine

### Model

- **YOLOv8m** (medium) — `yolov8m.pt`
- Auto-downloaded from Ultralytics on first run (~50 MB)
- Loaded once at startup: `model = YOLO("yolov8m.pt", task="detect")`

### Detection Parameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| `classes=[0]` | Person class only | Prevents cars, animals, objects from being detected |
| `conf=0.4` | 40% confidence threshold | Catches partially visible persons |
| `iou=0.4` | NMS IoU threshold | Reduces duplicate boxes on same person |

### Frame Processing Pipeline

```
Raw frame from IP cam
        │
        ▼
Resize to 640px width (YOLO optimal input size)
        │
        ▼
YOLOv8m inference (classes=[0], conf=0.4, iou=0.4)
        │
        ▼
Filter boxes: only class == 0 (person)
        │
        ▼
Draw bounding box + label (#1 85%, #2 91%...) per person
        │
        ▼
Draw "Persons: N" overlay (black outline + white text)
        │
        ▼
Store annotated frame in cameras[cam_id]["frame"]
        │
        ▼
If count changed AND count > 0 → INSERT into detections table
```

### Annotation Style

- Green bounding box (`RGB: 34, 197, 94`)
- Green filled label background with white text
- Label format: `#1  85%` (person number + confidence)
- Overlay text: `Persons: N` with black outline for readability on any background

---

## 7. Real-time Streaming

### Video Stream (`/video/<cam_id>`)

Uses **MJPEG over HTTP** — a multipart HTTP response where each part is a JPEG frame:

```
Content-Type: multipart/x-mixed-replace; boundary=frame

--frame
Content-Type: image/jpeg

<binary JPEG data>
--frame
...
```

- Frame rate: ~30 FPS (33ms sleep between frames)
- Thread-safe frame access via `threading.Lock`

### Live Updates (`/events`)

Uses **Server-Sent Events (SSE)** — a persistent HTTP connection where the server pushes text events:

```python
yield f"data: {cid}|{count}|{time}|{name}\n\n"
```

- Checks for changes every 500ms
- Only sends an event when the count or timestamp changes
- Browser reconnects automatically if connection drops

---

## 8. Excel Export

Route: `GET /export`

### File Structure

| Row | Content | Style |
|-----|---------|-------|
| 1 | "PersonMonitor — Detection Report" | Dark bg (#111827), white bold, size 16, merged A:D |
| 2 | Generated date/time | Light grey bg, grey text, merged A:D |
| 3 | Total Events (A:B) + Total Persons (C:D) | Grey bg, bold, split summary |
| 4 | Empty spacer | — |
| 5 | Column headers: #, Camera Name, Persons Detected, Timestamp | Dark bg (#1F2937), white bold, bordered |
| 6+ | Data rows | Alternating white/grey, person count in green bold |

### Additional Features

- Frozen pane at row 6 — headers stay visible when scrolling
- Thin borders on all data cells
- Filename: `detections_YYYYMMDD_HHMMSS.xlsx`
- Sorted oldest → newest (ascending by ID)

---

## 9. AI Assistant

### Overview

A floating chat button (bottom-right) is available on both the Dashboard and Admin Panel. It connects to Google Gemini 2.5 Flash via the `/ai` route and answers questions about the detection data.

### Route: `POST /ai`

**Request body:**
```json
{ "query": "Which camera had the most detections today?" }
```

**Response:**
```json
{ "answer": "The Front Door camera had the most detections with 42 events." }
```

### Role-based Data Context

The AI is given different data depending on the logged-in user's role:

| Data Provided to Gemini | Admin | User |
|---|---|---|
| Last 50 detections (cam, count, time) | ✅ | ✅ |
| Per-camera stats (events, total persons, last seen) | ✅ | ✅ |
| Total events and total persons logged | ✅ | ✅ |
| Active camera names and IP URLs | ✅ | ❌ |
| Registered users and their roles | ✅ | ❌ |

If a user asks about restricted data (camera IPs, user list), Gemini is instructed to respond that the information is restricted to admins.

### Configuration

- API key is loaded from `.env` file: `gemini_key = <your_key>`
- Model: `gemini-2.5-flash`
- Configured via `python-dotenv` and `google-generativeai`

### Example Questions

**User:**
- "How many persons were detected today?"
- "Which camera had the most activity?"
- "What was the last detection time?"

**Admin (additional):**
- "How many users are registered?"
- "Which cameras are currently active?"
- "Give me a summary of all camera activity"

---

## 10. Frontend Pages

### `login.html`

- Role toggle (User / Admin) using CSS radio buttons
- Selecting a role updates the page title, subtitle, and button text dynamically via JS
- Submits `expected_role` hidden field with the form
- Error message shown inline if credentials are wrong or role mismatches

### `dashboard.html`

- Sticky navigation bar with role badge and sign out
- Stats row: Total Events, Active Cameras, Live Persons (real-time)
- Camera feed grid: min 480px per card, live MJPEG stream embedded as `<img>`
- Per-camera person count badge — turns green when persons are detected
- Detection log: scrollable, max 30 entries, only shows `count > 0` events
- Download button (↓) in log header triggers `/export`
- SSE listener updates counts, badges, and log in real-time without page refresh
- Floating AI chat button (bottom-right) — opens panel, sends queries to `/ai`

### `admin.html`

- Stats row: Total Events, Cameras, Users
- Add Camera form: name + phone IP (URL built automatically as `http://<ip>:8080/video`)
- Add User form: username, password, role selector
- Active Cameras table with Remove button per camera
- Users table with Delete button (admin user protected)
- Detection Statistics table: events per camera + last detection time
- Toast notification on successful actions
- Floating AI chat button (bottom-right) — full admin data context

---

## 11. Configuration Reference

All configuration is in `app.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `app.secret_key` | `"monitor_secret_2024"` | Flask session signing key — change in production |
| `DB` | `"monitor.db"` | SQLite database filename |
| `model` | `YOLO("yolov8m.pt")` | Detection model — change to `yolov8l.pt` for higher accuracy |
| `conf=0.4` | `0.4` | Detection confidence threshold (0.0–1.0) |
| `iou=0.4` | `0.4` | NMS IoU threshold (0.0–1.0) |
| Camera port | `8080` | Built into URL as `http://<ip>:8080/video` |
| Default admin | `admin / admin123` | Created on first run if no admin exists |
| `gemini_key` | from `.env` | Google Gemini API key for AI assistant |

### Tuning Detection Accuracy

| Goal | Change |
|------|--------|
| Detect more people (fewer misses) | Lower `conf` to `0.3` |
| Reduce false detections | Raise `conf` to `0.5` or `0.6` |
| Better accuracy overall | Switch to `yolov8l.pt` or `yolov8x.pt` |
| Faster processing (weaker hardware) | Switch to `yolov8s.pt` |

---

## 12. Troubleshooting

### `OSError: DLL initialization failed` on startup

PyTorch does not support Python 3.13 or 3.14. Use Python 3.11 or 3.12.

```bash
py -3.11 -m venv venv
```

### Camera feed shows blank / not loading

- Confirm phone and PC are on the same Wi-Fi network
- Open `http://<phone-ip>:8080` in a browser to verify the stream is live
- Check the IP entered in Admin Panel is correct (no `http://`, no port)

### No detections appearing

- Ensure good lighting — YOLO struggles in very dark environments
- Move the camera closer or adjust angle so full/partial body is visible
- Lower confidence threshold: change `conf=0.4` to `conf=0.3` in `camera_worker()`

### Database errors on startup

Delete `monitor.db` and restart — it will be recreated with the correct schema:

```bash
del monitor.db
python app.py
```

### Export downloads empty file

No detections have been recorded yet. The export only includes rows where `count > 0`.

### AI Assistant not responding

- Verify `.env` file exists in the project root with `gemini_key = <your_key>`
- Ensure `google-generativeai` and `python-dotenv` are installed
- Check your Gemini API key is valid and has quota remaining
