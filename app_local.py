import io
import os
import base64
import re
import numpy as np
import cv2
import rasterio
from rasterio.windows import from_bounds as window_from_bounds
from rasterio.warp import transform_bounds
from rasterio.transform import from_bounds
from flask import Flask, request, jsonify, render_template, send_file
from pystac_client import Client
import planetary_computer
from PIL import Image

app = Flask(__name__, template_folder='templates')

SAVE_DIR = os.path.abspath("saved_images")
os.makedirs(SAVE_DIR, exist_ok=True)

catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

def fetch_optimized_imagery(bbox, date_range, label="image"):
    """
    Fetch Sentinel-2 imagery using direct rasterio windowed reading in native UTM CRS.
    Guarantees lightning-fast loading and prevents odc.stac hanging.
    """
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
        # Transform WGS84 bbox [min_lon, min_lat, max_lon, max_lat] to the item's native UTM CRS
        native_bounds = transform_bounds("EPSG:4326", native_crs, min_lon, min_lat, max_lon, max_lat)
        
        window = window_from_bounds(*native_bounds, transform=src.transform)
        
        # Read bands using native window
        with rasterio.open(red_url) as r_src, \
             rasterio.open(green_url) as g_src, \
             rasterio.open(blue_url) as b_src:
             
            red = r_src.read(1, window=window, boundless=True, fill_value=0).astype(np.float32)
            green = g_src.read(1, window=window, boundless=True, fill_value=0).astype(np.float32)
            blue = b_src.read(1, window=window, boundless=True, fill_value=0).astype(np.float32)

    rgb = np.stack([red, green, blue], axis=-1)
    rgb = np.nan_to_num(rgb, nan=0.0)

    # If window is too small, pad or force a minimum dimension
    if rgb.shape[0] < 10 or rgb.shape[1] < 10:
        print("-> Warning: Region is very small. Expanding crop window...")
        with rasterio.open(red_url) as r_src:
            center_x, center_y = (native_bounds[0] + native_bounds[2]) / 2, (native_bounds[1] + native_bounds[3]) / 2
            half_size = 50  # 50 pixels each way (~500m)
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

    # Scale reflectance values to uint8 [0, 255]
    rgb = np.clip((rgb / 3000.0) * 255, 0, 255).astype(np.uint8)
    print(f"-> Successfully loaded image array of shape: {rgb.shape}")

    # Completely sanitize STAC ID for Windows filesystem compatibility
    safe_id = re.sub(r'[^a-zA-Z0-9_-]', '_', str(selected_item.id))
    filename = os.path.join(SAVE_DIR, f"{label}_{safe_id}.png")
    
    pil_save = Image.fromarray(rgb)
    pil_save.save(filename)
    print(f"-> Saved full scene image locally to: {filename}")

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


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/api/estimate', methods=['POST'])
def estimate():
    return jsonify({"estimated_seconds": 10})


@app.route('/api/detect-satellite', methods=['POST'])
def detect_satellite():
    data = request.json
    if not data:
        return jsonify({"status": "error", "error": "No payload provided"}), 400

    bbox = data.get('bbox')
    t1_date = data.get('t1_date')
    t2_date = data.get('t2_date')

    try:
        print("\n==============================\nRUNNING CHANGE DETECTION PIPELINE\n==============================")
        
        print("\n[1/3] Fetching Time 1 (Baseline) Imagery...")
        t1_rgb, transform = fetch_optimized_imagery(bbox, t1_date, label="T1_baseline")

        print("\n[2/3] Fetching Time 2 (Current) Imagery...")
        t2_rgb, _ = fetch_optimized_imagery(bbox, t2_date, label="T2_current")

        if t1_rgb.shape != t2_rgb.shape:
            print("-> Resizing T2 image to match T1 dimensions...")
            t2_rgb = cv2.resize(t2_rgb, (t1_rgb.shape[1], t1_rgb.shape[0]))

        print("\n[3/3] Running change detection contours...")
        diff = cv2.absdiff(t1_rgb, t2_rgb)
        gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"-> Found {len(contours)} raw change regions.")

        events = []
        event_id = 1

        for cnt in contours:
            area_pixels = cv2.contourArea(cnt)
            if area_pixels < 25:  
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            center_x, center_y = x + w // 2, y + h // 2
            lat, lon = pixel_to_latlon(center_x, center_y, transform)

            t1_patch = crop_to_base64(t1_rgb, [x, y, w, h])
            t2_patch = crop_to_base64(t2_rgb, [x, y, w, h])

            area_sq_m = int(area_pixels * 100)
            confidence = round(float(np.min([98.5, 65.0 + (area_pixels / 10.0)])), 1)
            
            if area_sq_m > 10000:
                severity, activity_type = "HIGH", "Illegal Construction / Urban Expansion"
            elif area_sq_m > 3000:
                severity, activity_type = "MEDIUM", "Land Clearing / Earthwork"
            else:
                severity, activity_type = "LOW", "Minor Structural Change"

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

        print(f"-> Successfully processed {len(events)} significant events.")
        return jsonify({
            "status": "success",
            "events_count": len(events),
            "events": events
        })

    except Exception as e:
        print(f"[ERROR] Change detection failed: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)