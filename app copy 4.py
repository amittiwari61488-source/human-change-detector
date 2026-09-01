# --- FIX FOR NUMPY 2.0+ & TENSORBOARD/ACCELERATE CONFLICT ---
import numpy as np
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_
# -----------------------------------------------------------

import io
import os
import json
import base64
import re
import time
import cv2
import rasterio
from rasterio.enums import Resampling
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

# Advanced Data Processing Libraries
from skimage.exposure import match_histograms
from shapely.geometry import Polygon, mapping

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

UPLOAD_FOLDER = os.path.abspath("uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- STAC CATALOG SETUP (MAP MODE) ---
catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

# --- AI MODEL SETUP ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join('model', 'best_siamese_model.pth')
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

# In-memory store for the latest detected events (Shared for PDF, Copilot & GeoJSON)
latest_events = []

# Initialize Groq Client
client = Groq()

# ==========================================
# ADVANCED UTILITY FUNCTIONS
# ==========================================
def calculate_ndvi(nir, red):
    """Calculates Normalized Difference Vegetation Index"""
    # Avoid division by zero
    denominator = (nir + red)
    denominator[denominator == 0] = 1e-10
    return (nir - red) / denominator

def fetch_optimized_imagery(bbox, date_range, label="image"):
    print(f"\n--- Searching STAC Catalog ---")
    print(f"BBox (WGS84): {bbox}")
    print(f"Date Range: {date_range}")

    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": 20}},  # Increased slightly because we use true cloud masking now
        sortby=["eo:cloud_cover"]
    )
    
    items = list(search.items())
    if not items:
        raise ValueError(f"No Sentinel-2 imagery found for date range: {date_range}")

    selected_item = items[0]
    print(f"-> Selected best image ID: {selected_item.id} (Cloud cover: {selected_item.properties.get('eo:cloud_cover')}%)")

    # RGB & Multi-spectral Bands
    red_url = selected_item.assets["B04"].href
    green_url = selected_item.assets["B03"].href
    blue_url = selected_item.assets["B02"].href
    nir_url = selected_item.assets["B08"].href  # Near Infrared (10m)
    scl_url = selected_item.assets["SCL"].href  # Scene Classification Layer for Cloud Masking (20m)

    min_lon, min_lat, max_lon, max_lat = bbox

    with rasterio.open(red_url) as src:
        native_crs = src.crs
        native_bounds = transform_bounds("EPSG:4326", native_crs, min_lon, min_lat, max_lon, max_lat)
        window = window_from_bounds(*native_bounds, transform=src.transform)
        
        # Read 10m bands (RGB + NIR)
        with rasterio.open(red_url) as r_src, rasterio.open(green_url) as g_src, rasterio.open(blue_url) as b_src, rasterio.open(nir_url) as nir_src:
            red = r_src.read(1, window=window, boundless=True, fill_value=0).astype(np.float32)
            green = g_src.read(1, window=window, boundless=True, fill_value=0).astype(np.float32)
            blue = b_src.read(1, window=window, boundless=True, fill_value=0).astype(np.float32)
            nir = nir_src.read(1, window=window, boundless=True, fill_value=0).astype(np.float32)
            
            out_shape = red.shape

        # Read 20m SCL band and resample to match 10m output shape
        with rasterio.open(scl_url) as scl_src:
            scl_window = window_from_bounds(*native_bounds, transform=scl_src.transform)
            scl = scl_src.read(1, window=scl_window, out_shape=out_shape, resampling=Resampling.nearest)

    rgb = np.stack([red, green, blue], axis=-1)
    rgb = np.nan_to_num(rgb, nan=0.0)
    
    # Calculate NDVI
    ndvi = calculate_ndvi(nir, red)

    # Normalize RGB for AI input
    rgb = np.clip((rgb / 3000.0) * 255, 0, 255).astype(np.uint8)
    
    height, width, _ = rgb.shape
    transform = from_bounds(min_lon, min_lat, max_lon, max_lat, width, height)

    return rgb, ndvi, scl, transform

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

def contour_to_geojson_polygon(contour, transform):
    """Converts OpenCV contour pixels to a GeoJSON mapping polygon"""
    points = []
    for point in contour:
        x, y = point[0]
        lon, lat = pixel_to_latlon(x, y, transform)
        points.append((lon, lat))
    
    # Close the polygon
    if len(points) >= 3:
        points.append(points[0])
        return mapping(Polygon(points))
    return None

# ==========================================
# FLASK ROUTES
# ==========================================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/saved_images/<path:filename>')
def serve_saved_images(filename):
    return send_from_directory(SAVE_DIR, filename)

@app.route('/uploads/<path:filename>')
def serve_uploads(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# --- MAP MODE ENDPOINT (ADVANCED AI PIPELINE) ---
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
        if model is None:
            return jsonify({"status": "error", "error": "AI Model not loaded."}), 500

        # 1. Fetch Images, Indices (NDVI), and Cloud Masks (SCL)
        t1_rgb, t1_ndvi, t1_scl, transform = fetch_optimized_imagery(bbox, t1_date, label="T1_baseline")
        t2_rgb, t2_ndvi, t2_scl, _ = fetch_optimized_imagery(bbox, t2_date, label="T2_current")

        if t1_rgb.shape != t2_rgb.shape:
            t2_rgb = cv2.resize(t2_rgb, (t1_rgb.shape[1], t1_rgb.shape[0]))
            t2_ndvi = cv2.resize(t2_ndvi, (t1_rgb.shape[1], t1_rgb.shape[0]))
            t2_scl = cv2.resize(t2_scl, (t1_rgb.shape[1], t1_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

        # 2. Histogram Matching (Fixes seasonal color variations & shadows)
        t2_rgb_matched = match_histograms(t2_rgb, t1_rgb, channel_axis=-1)

        # 3. Save arrays temporarily for Model Inference
        timestamp = int(time.time())
        sat_t1_path = os.path.join(SAVE_DIR, f"sat_t1_{timestamp}.png")
        sat_t2_path = os.path.join(SAVE_DIR, f"sat_t2_{timestamp}.png")
        sat_mask_path = os.path.join(SAVE_DIR, f"sat_mask_{timestamp}.png")

        Image.fromarray(t1_rgb).save(sat_t1_path)
        Image.fromarray(t2_rgb_matched).save(sat_t2_path) # Pass matched image to model

        # 4. Predict changes using Siamese Model
        run_model_inference(model, sat_t1_path, sat_t2_path, sat_mask_path, device=DEVICE)

        # 5. Load the generated AI mask
        mask_img = cv2.imread(sat_mask_path, cv2.IMREAD_GRAYSCALE)
        
        # 6. Apply True Cloud Masking (SCL values: 3=Cloud shadows, 8=Cloud medium prob, 9=Cloud high prob, 10=Cirrus)
        cloud_classes = [3, 8, 9, 10]
        cloud_mask = np.isin(t1_scl, cloud_classes) | np.isin(t2_scl, cloud_classes)
        mask_img[cloud_mask] = 0 # Force mask to 0 (No Change) where clouds exist

        # Threshold and morphology
        _, thresh = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # 7. Extract Events & Semantics
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
            
            # Generate GeoJSON Polygon geometry for export
            geo_polygon = contour_to_geojson_polygon(cnt, transform)

            # Analyze multi-spectral data inside the contour
            contour_mask = np.zeros(mask_img.shape, dtype=np.uint8)
            cv2.drawContours(contour_mask, [cnt], -1, 255, -1)
            
            mean_ndvi_t1 = np.mean(t1_ndvi[contour_mask == 255])
            mean_ndvi_t2 = np.mean(t2_ndvi[contour_mask == 255])

            area_sq_m = int(area_pixels * 100)
            confidence = round(float(np.min([98.5, 65.0 + (area_pixels / 10.0)])), 1)
            
            # Advanced Semantic AI Classification based on NDVI changes
            if mean_ndvi_t1 > 0.3 and mean_ndvi_t2 < 0.15:
                activity_type = "Deforestation / Land Clearing"
                severity = "HIGH"
            elif mean_ndvi_t1 < 0.1 and mean_ndvi_t2 > 0.3:
                activity_type = "New Vegetation / Afforestation"
                severity = "LOW"
            elif area_sq_m > 5000:
                activity_type = "Large Infrastructure Construction"
                severity = "HIGH"
            elif area_sq_m > 2000:
                activity_type = "Structure Assembly"
                severity = "MEDIUM"
            else:
                activity_type = "Minor Terrain Change"
                severity = "LOW"

            t1_patch = crop_to_base64(t1_rgb, [x, y, w, h])
            t2_patch = crop_to_base64(t2_rgb, [x, y, w, h])

            events.append({
                "id": event_id,
                "lat": lat,
                "lon": lon,
                "activity_type": activity_type,
                "severity": severity,
                "confidence": confidence,
                "area_sq_m": area_sq_m,
                "t1_patch": t1_patch,
                "t2_patch": t2_patch,
                "geometry": geo_polygon
            })
            event_id += 1

        t1_full_name = f"annotated_t1_{timestamp}.png"
        t2_full_name = f"annotated_t2_{timestamp}.png"
        
        Image.fromarray(t1_annotated).save(os.path.join(SAVE_DIR, t1_full_name))
        Image.fromarray(t2_annotated).save(os.path.join(SAVE_DIR, t2_full_name))

        latest_events = events
        
        return jsonify({
            "status": "success",
            "events_count": len(events),
            "events": events,
            "t1_full_url": f"/saved_images/{t1_full_name}",
            "t2_full_url": f"/saved_images/{t2_full_name}",
            "mask_url": f"/saved_images/sat_mask_{timestamp}.png"
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
    
    latest_events = events
    timestamp = int(time.time())
    
    return jsonify({
        "status": "success",
        "events_count": len(events),
        "annotated_image": f"/uploads/{os.path.basename(annotated_path)}?t={timestamp}",
        "mask_image": f"/uploads/predicted_mask.png?t={timestamp}",
        "events": events
    })

# --- DATA EXPORT ENDPOINTS ---
@app.route('/api/download-geojson', methods=['GET'])
def download_geojson():
    """Generates standard GeoJSON from detected events for GIS software."""
    global latest_events
    
    features = []
    for event in latest_events:
        if event.get('geometry'):
            feature = {
                "type": "Feature",
                "properties": {
                    "id": event.get("id"),
                    "activity_type": event.get("activity_type"),
                    "severity": event.get("severity"),
                    "confidence": event.get("confidence"),
                    "area_sq_m": event.get("area_sq_m")
                },
                "geometry": event.get("geometry")
            }
            features.append(feature)
            
    feature_collection = {
        "type": "FeatureCollection",
        "features": features
    }
    
    return jsonify(feature_collection)

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