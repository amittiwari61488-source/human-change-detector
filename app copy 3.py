# --- FIX FOR NUMPY 2.0+ & TENSORBOARD/ACCELERATE CONFLICT ---
import numpy as np
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_
# -----------------------------------------------------------

import io
import os
import base64
import re
import time
import cv2
import rasterio
import torch
import sys
from datetime import datetime
from fpdf import FPDF
from groq import Groq
from rasterio.windows import from_bounds as window_from_bounds
from rasterio.warp import transform_bounds
from rasterio.transform import from_bounds
from flask import Flask, request, jsonify, render_template, send_file, send_from_directory
from pystac_client import Client
import planetary_computer
from PIL import Image

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

# Add root src directory to python path for Manual mode dependencies
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.model import SiameseUNetAttention
    from src.inference import run_model_inference
    from src.post_process import process_detected_changes
except ImportError:
    print("Warning: Manual mode src modules not found. Ensure src directory is accessible.")

app = Flask(__name__, template_folder='templates')

# --- DIRECTORIES SETUP ---
SAVE_DIR = os.path.abspath("saved_images")
os.makedirs(SAVE_DIR, exist_ok=True)

# FIX: Isolated uploads folder to prevent Flask static routing conflicts
UPLOAD_FOLDER = os.path.abspath("uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- STAC CATALOG SETUP (MAP MODE) ---
catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

# --- AI MODEL SETUP (MANUAL MODE) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join('models', 'best_siamese_model.pth')
model = None

print("🚀 Initializing AI Change Detection Model for Flask...")
try:
    model = SiameseUNetAttention(pretrained=False).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        print("✅ Model weights loaded successfully!")
    else:
        print(f"⚠️ Warning: Model weights not found at {MODEL_PATH}.")
except Exception as e:
    print(f"⚠️ Model Initialization Error: {e}")

# In-memory store for the latest detected events (Shared for PDF & Copilot)
latest_events = []

# Initialize Groq Client
client = Groq()

# ==========================================
# UTILITY FUNCTIONS
# ==========================================
def fetch_optimized_imagery(bbox, date_range, label="image"):
    print(f"\n--- Searching STAC Catalog ---")
    print(f"BBox (WGS84): {bbox}")
    print(f"Date Range: {date_range}")

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": 15}},
        sortby=["eo:cloud_cover"]
    )
    
    items = list(search.items())
    print(f"-> Total images found: {len(items)}")

    if not items:
        raise ValueError(f"No Sentinel-2 imagery found for date range: {date_range}")

    selected_item = items[0]
    print(f"-> Selected best image ID: {selected_item.id} (Cloud cover: {selected_item.properties.get('eo:cloud_cover')}%)")

    red_url = selected_item.assets["B04"].href
    green_url = selected_item.assets["B03"].href
    blue_url = selected_item.assets["B02"].href

    min_lon, min_lat, max_lon, max_lat = bbox

    with rasterio.open(red_url) as src:
        native_crs = src.crs
        native_bounds = transform_bounds("EPSG:4326", native_crs, min_lon, min_lat, max_lon, max_lat)
        window = window_from_bounds(*native_bounds, transform=src.transform)
        
        with rasterio.open(red_url) as r_src, \
             rasterio.open(green_url) as g_src, \
             rasterio.open(blue_url) as b_src:
             
            red = r_src.read(1, window=window, boundless=True, fill_value=0).astype(np.float32)
            green = g_src.read(1, window=window, boundless=True, fill_value=0).astype(np.float32)
            blue = b_src.read(1, window=window, boundless=True, fill_value=0).astype(np.float32)

    rgb = np.stack([red, green, blue], axis=-1)
    rgb = np.nan_to_num(rgb, nan=0.0)

    if rgb.shape[0] < 10 or rgb.shape[1] < 10:
        print("-> Warning: Region is very small. Expanding crop window...")
        with rasterio.open(red_url) as r_src:
            center_x, center_y = (native_bounds[0] + native_bounds[2]) / 2, (native_bounds[1] + native_bounds[3]) / 2
            half_size = 50 
            win = rasterio.windows.Window(
                r_src.index(center_x, center_y)[1] - half_size,
                r_src.index(center_x, center_y)[0] - half_size,
                half_size * 2,
                half_size * 2
            )
            red = r_src.read(1, window=win, boundless=True, fill_value=0).astype(np.float32)
        with rasterio.open(green_url) as g_src:
            green = g_src.read(1, window=win, boundless=True, fill_value=0).astype(np.float32)
        with rasterio.open(blue_url) as b_src:
            blue = b_src.read(1, window=win, boundless=True, fill_value=0).astype(np.float32)
        rgb = np.stack([red, green, blue], axis=-1)
        rgb = np.nan_to_num(rgb, nan=0.0)

    rgb = np.clip((rgb / 3000.0) * 255, 0, 255).astype(np.uint8)
    
    safe_id = re.sub(r'[^a-zA-Z0-9_-]', '_', str(selected_item.id))
    filename = os.path.join(SAVE_DIR, f"{label}_{safe_id}.png")
    
    Image.fromarray(rgb).save(filename)
    
    height, width, _ = rgb.shape
    transform = from_bounds(min_lon, min_lat, max_lon, max_lat, width, height)

    return rgb, transform

def pixel_to_latlon(x, y, transform):
    lon, lat = transform * (x, y)
    return float(lat), float(lon)

def crop_to_base64(img_array, bbox_crop):
    x, y, w, h = bbox_crop
    pad = 20
    h_img, w_img, _ = img_array.shape
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(w_img, x + w + pad), min(h_img, y + h + pad)

    cropped = img_array[y1:y2, x1:x2]
    pil_img = Image.fromarray(cropped)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()

# --- PDF REPORT GENERATOR CLASS ---
class PDFReport(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 16)
        self.set_text_color(33, 37, 41)
        self.cell(0, 10, 'Change Detection & Semantic AI Audit Report', 0, 1, 'CENTER')
        
        self.set_font('helvetica', '', 10)
        self.set_text_color(108, 117, 125)
        self.cell(0, 6, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'CENTER')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.set_text_color(108, 117, 125)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'CENTER')


# ==========================================
# FLASK ROUTES
# ==========================================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/saved_images/<path:filename>')
def serve_saved_images(filename):
    return send_from_directory(SAVE_DIR, filename)

# FIX: Update route to match the new isolated folder
@app.route('/uploads/<path:filename>')
def serve_uploads(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/estimate', methods=['POST'])
def estimate():
    return jsonify({"estimated_seconds": 10})

# --- MAP MODE ENDPOINT ---
@app.route('/api/detect-satellite', methods=['POST'])
def detect_satellite():
    global latest_events
    data = request.json
    if not data:
        return jsonify({"status": "error", "error": "No payload provided"}), 400

    bbox = data.get('bbox')
    t1_date = data.get('t1_date')
    t2_date = data.get('t2_date')

    try:
        t1_rgb, transform = fetch_optimized_imagery(bbox, t1_date, label="T1_baseline")
        t2_rgb, _ = fetch_optimized_imagery(bbox, t2_date, label="T2_current")

        if t1_rgb.shape != t2_rgb.shape:
            t2_rgb = cv2.resize(t2_rgb, (t1_rgb.shape[1], t1_rgb.shape[0]))

        diff = cv2.absdiff(t1_rgb, t2_rgb)
        gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        t1_annotated = t1_rgb.copy()
        t2_annotated = t2_rgb.copy()

        events = []
        event_id = 1

        for cnt in contours:
            area_pixels = cv2.contourArea(cnt)
            if area_pixels < 25:  
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(t1_annotated, (x, y), (x + w, y + h), (255, 0, 0), 3)
            cv2.rectangle(t2_annotated, (x, y), (x + w, y + h), (255, 0, 0), 3)

            center_x, center_y = x + w // 2, y + h // 2
            lat, lon = pixel_to_latlon(center_x, center_y, transform)

            t1_patch = crop_to_base64(t1_rgb, [x, y, w, h])
            t2_patch = crop_to_base64(t2_rgb, [x, y, w, h])

            area_sq_m = int(area_pixels * 100)
            confidence = round(float(np.min([98.5, 65.0 + (area_pixels / 10.0)])), 1)
            
            if area_sq_m > 10000:
                severity, activity_type = "HIGH", "Large Construction"
            elif area_sq_m > 3000:
                severity, activity_type = "MEDIUM", "Land Clearing"
            else:
                severity, activity_type = "LOW", "Minor Change"

            events.append({
                "id": event_id,
                "lat": lat,
                "lon": lon,
                "activity_type": activity_type,
                "severity": severity,
                "confidence": confidence,
                "area_sq_m": area_sq_m,
                "t1_patch": t1_patch,
                "t2_patch": t2_patch
            })
            event_id += 1

        timestamp = int(time.time())
        t1_full_name = f"annotated_t1_{timestamp}.png"
        t2_full_name = f"annotated_t2_{timestamp}.png"
        
        Image.fromarray(t1_annotated).save(os.path.join(SAVE_DIR, t1_full_name))
        Image.fromarray(t2_annotated).save(os.path.join(SAVE_DIR, t2_full_name))

        # Update global events for Copilot / PDF
        latest_events = events
        
        return jsonify({
            "status": "success",
            "events_count": len(events),
            "events": events,
            "t1_full_url": f"/saved_images/{t1_full_name}",
            "t2_full_url": f"/saved_images/{t2_full_name}"
        })

    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


# --- MANUAL MODE ENDPOINT ---
@app.route('/api/detect', methods=['POST'])
def detect_changes():
    global latest_events
    if 'pre_image' not in request.files or 'post_image' not in request.files:
        return jsonify({"error": "Please provide both pre and post change images."}), 400
        
    t1_file = request.files['pre_image']
    t2_file = request.files['post_image']
    
    t1_path = os.path.join(app.config['UPLOAD_FOLDER'], 't1.png')
    t2_path = os.path.join(app.config['UPLOAD_FOLDER'], 't2.png')
    mask_path = os.path.join(app.config['UPLOAD_FOLDER'], 'predicted_mask.png')
    
    t1_file.save(t1_path)
    t2_file.save(t2_path)
    
    run_model_inference(model, t1_path, t2_path, mask_path, device=DEVICE)
    
    events, annotated_path = process_detected_changes(
        mask_path=mask_path, 
        post_image_path=t2_path, 
        output_dir=app.config['UPLOAD_FOLDER']
    )
    
    # Save events globally for PDF export & Copilot context
    latest_events = events
    timestamp = int(time.time())
    
    # FIX: Return formatted image paths using the new /uploads/ route
    return jsonify({
        "status": "success",
        "events_count": len(events),
        "annotated_image": f"/uploads/{os.path.basename(annotated_path)}?t={timestamp}",
        "mask_image": f"/uploads/predicted_mask.png?t={timestamp}",
        "events": events
    })


# --- PDF & COPILOT ENDPOINTS ---
@app.route('/api/download-pdf', methods=['GET'])
def download_pdf():
    global latest_events
    
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    pdf.set_font('helvetica', 'B', 12)
    pdf.set_text_color(33, 37, 41)
    pdf.cell(0, 8, 'Executive Summary', 0, 1)
    
    pdf.set_font('helvetica', '', 10)
    pdf.multi_cell(0, 6, 
        f"Total Violations / Events Detected: {len(latest_events)}\n"
        "This document summarizes the semantic change classification results "
        "and spatial expansion metrics automatically flagged by the monitoring pipeline."
    )
    pdf.ln(5)
    
    pdf.set_font('helvetica', 'B', 12)
    pdf.cell(0, 8, 'Detected Violation Events', 0, 1)
    
    pdf.set_font('helvetica', 'B', 10)
    pdf.set_fill_color(220, 224, 230)
    pdf.cell(15, 8, 'ID', 1, 0, 'CENTER', True)
    pdf.cell(75, 8, 'Activity Type (Semantic AI)', 1, 0, 'LEFT', True)
    pdf.cell(35, 8, 'Area (m²)', 1, 0, 'CENTER', True)
    pdf.cell(30, 8, 'Confidence', 1, 0, 'CENTER', True)
    pdf.cell(35, 8, 'Severity', 1, 1, 'CENTER', True)
    
    pdf.set_font('helvetica', '', 9)
    for event in latest_events:
        event_id = str(event.get('id', event.get('ID', '-')))
        event_type = str(event.get('activity_type', event.get('Type', event.get('Activity Type', '-'))))
        event_area = str(event.get('area_sq_m', event.get('Area', '-')))
        event_conf = str(event.get('confidence', event.get('Confidence', '-')))
        event_sev = str(event.get('severity', event.get('Severity', '-')))

        pdf.cell(15, 8, event_id, 1, 0, 'CENTER')
        pdf.cell(75, 8, event_type, 1, 0, 'LEFT')
        pdf.cell(35, 8, event_area, 1, 0, 'CENTER')
        pdf.cell(30, 8, event_conf, 1, 0, 'CENTER')
        pdf.cell(35, 8, event_sev, 1, 1, 'CENTER')
        
    pdf_bytes = pdf.output()
    pdf_buffer = io.BytesIO(bytes(pdf_bytes))
    
    return send_file(
        pdf_buffer,
        as_attachment=True,
        download_name=f"change_detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mimetype='application/pdf'
    )

@app.route('/api/copilot', methods=['POST'])
def copilot_query():
    global latest_events
    data = request.get_json()
    user_query = data.get('query', '').strip()

    if not latest_events:
        return jsonify({
            "response": "⚠️ Please run a change detection pipeline (Map or Manual) first so I have data context to analyze!"
        })

    if not os.environ.get("GROQ_API_KEY"):
        return jsonify({
            "response": "⚠️ GROQ_API_KEY is not set in your .env file. Please add it and restart the server."
        })

    context_str = f"Current Change Detection & Violation Events Summary ({len(latest_events)} events detected):\n"
    for e in latest_events:
        context_str += (
            f"- ID: #{e.get('id', e.get('ID', '-'))}, "
            f"Activity Type: {e.get('activity_type', e.get('Type', '-'))}, "
            f"Area: {e.get('area_sq_m', e.get('Area', '-'))} m², "
            f"Confidence: {e.get('confidence', e.get('Confidence', '-'))}%, "
            f"Severity: {e.get('severity', e.get('Severity', '-'))}\n"
        )

    system_prompt = (
        "You are an expert Geospatial AI Compliance Copilot. "
        "Your job is to assist users in analyzing satellite change detection and human activity violation events. "
        "Answer the user's question clearly, professionally, and concisely based strictly on the provided data context below. "
        "Do not make up facts outside this context.\n\n"
        f"Data Context:\n{context_str}"
    )

    try:
        completion = client.chat.completions.create(
            model="openai/gpt-oss-20b",  
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            temperature=0.3,
            max_tokens=600,
        )
        bot_reply = completion.choices[0].message.content
        return jsonify({"response": bot_reply})

    except Exception as e:
        return jsonify({"response": f"❌ Error connecting to Groq API: {str(e)}"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)