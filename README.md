# PersonMonitor

AI-powered real-time person detection and monitoring system using a phone IP camera, YOLOv8, Flask, and SQLite.

---

## Features

- Real-time person detection via phone IP camera stream
- YOLOv8m model — detects multiple persons simultaneously with confidence scores
- Role-based login — Admin and User with separate access levels
- Admin panel — add/remove cameras and users
- Live dashboard — real-time video feed with person count
- Detection log — only logs events when persons are detected
- Export detection history as a structured `.xlsx` report
- AI Assistant — floating chat button powered by Gemini 2.5 Flash
- Role-restricted AI — admin sees full system data, user sees detection data only
- All data stored locally in SQLite

---

## Project Structure

```
monitering/
├── app.py                  # Flask application — core logic
├── requirements.txt        # Python dependencies
├── monitor.db              # SQLite database (auto-created)
├── yolov8m.pt              # YOLOv8 medium model (auto-downloaded)
├── README.md
├── DOCUMENTATION.md
└── templates/
    ├── login.html          # Login page (User / Admin toggle)
    ├── dashboard.html      # Live monitoring dashboard
    └── admin.html          # Admin management panel
```

---

## Requirements

- Python 3.11 or 3.12 (PyTorch does not support Python 3.14+)
- Phone with IP Webcam app installed
  - Android: [IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam)
  - iOS: [EpocCam](https://apps.apple.com/app/epoccam-webcam-for-mac-and-pc/id449133483)

---

## Installation

```bash
# 1. Create virtual environment with Python 3.11
py -3.11 -m venv venv

# 2. Activate
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Running

```bash
python app.py
```

Open your browser at: `http://localhost:5000`

The YOLOv8n model (`yolov8n.pt`) will auto-download on first run (~6 MB).

> Model weights (`.pt` files) are not included in the repository. They are automatically downloaded by Ultralytics on first run.

---

## Default Login

| Role  | Username | Password  |
|-------|----------|-----------|
| Admin | admin    | admin123  |

> Admin can create additional users from the Admin Panel.

---

## Usage

### Admin
1. Login with Admin tab selected
2. Go to **Admin Panel**
3. Add a camera — enter a name and the phone's IP address (e.g. `192.168.1.5`)
4. Add users with User or Admin role
5. View detection statistics per camera

### User
1. Login with User tab selected
2. View live camera feeds on the dashboard
3. Monitor real-time person count and detection log
4. Download detection history as Excel report

---

## Phone Camera Setup

1. Install **IP Webcam** on your Android phone
2. Open the app → scroll down → tap **Start server**
3. Note the IP address shown (e.g. `192.168.1.5`)
4. Make sure your phone and PC are on the same Wi-Fi network
5. Enter that IP in the Admin Panel — the app connects to `http://<ip>:8080/video`

---

## Excel Export

Click the download icon (↓) in the Detection Log header to download a `.xlsx` file containing:

- Report title and generated timestamp
- Summary: total events and total persons logged
- Full detection history with camera name, person count, and timestamp
- Alternating row colors, frozen header row

---

## Tech Stack

| Component     | Technology          |
|---------------|---------------------|
| Backend       | Flask (Python)      |
| Detection     | YOLOv8m (Ultralytics) |
| Video capture | OpenCV              |
| Database      | SQLite              |
| Real-time     | Server-Sent Events (SSE) |
| Export        | openpyxl            |
| AI Assistant  | Gemini 2.5 Flash (Google) |
| Frontend      | HTML, CSS, Vanilla JS |
