#!/usr/bin/env python3
"""
PZT Compare v2 - Flask backend
Port: 8899
"""

import os
import json
import uuid
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import cv2
from functools import wraps
from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory, Response
from werkzeug.utils import secure_filename
import subprocess
import hashlib, hmac
import openai
import base64
import time

app = Flask(__name__)

# --- Password-only Auth (cookie-based) ---
APP_PASSWORD = os.environ.get('PZT_PASS', 'PI2026!')
app.secret_key = os.environ.get('SECRET_KEY', 'pzt-compare-secret-key-2026')

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        from flask import session as flask_session
        if not flask_session.get('authenticated'):
            return redirect(url_for('login_page', next=request.path))
        return f(*args, **kwargs)
    return decorated

@app.before_request
def before_request_auth():
    from flask import session as flask_session
    # Skip auth for health check and login page
    if request.path in ('/health', '/login') or request.path.startswith('/static'):
        return
    if not flask_session.get('authenticated'):
        return redirect(url_for('login_page', next=request.path))

@app.route('/login', methods=['GET', 'POST'])
def login_page():
    from flask import session as flask_session
    error = None
    if request.method == 'POST':
        password = request.form.get('password', '')
        if password == APP_PASSWORD:
            flask_session['authenticated'] = True
            next_url = request.args.get('next', '/')
            return redirect(next_url)
        else:
            error = 'Nieprawidłowe hasło'
    
    return f'''<!DOCTYPE html>
<html lang="pl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PZT Compare — Logowanie</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>📐</text></svg>">
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root {{ 
    --navy: #003d6c; --orange: #f58220; --green: #6bb98f; --red: #e01e20;
    --bg: #f5f5f5; --card: #ffffff; --border: #e0e0e0; --text: #262626; --muted: #636363;
  }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:'Poppins',-apple-system,sans-serif; 
         background: linear-gradient(135deg, #003d6c 0%, #28486e 100%); color:var(--text);
         display:flex; align-items:center; justify-content:center; min-height:100vh; }}
  .login-box {{ background:var(--card); border:1px solid rgba(255,255,255,0.1); border-radius:16px;
               padding:2.5rem; width:100%; max-width:380px; text-align:center; 
               box-shadow: 0 8px 32px rgba(0,0,0,0.3); }}
  .logo {{ margin-bottom: 1.5rem; }}
  .logo img {{ height: 60px; object-fit: contain; }}
  .login-box h1 {{ font-size:1.8rem; margin-bottom:.5rem; color:var(--navy); font-weight:600; }}
  .login-box p {{ color:var(--muted); font-size:.95rem; margin-bottom:2rem; }}
  .login-box input {{
    width:100%; padding:.8rem 1rem; border:1px solid var(--border); border-radius:8px;
    background:var(--card); color:var(--text); font-size:1rem; text-align:center;
    letter-spacing:2px; margin-bottom:1rem; transition: border-color 0.2s;
  }}
  .login-box input:focus {{ outline:none; border-color:var(--navy); }}
  .login-box button {{
    width:100%; padding:.9rem; border:none; border-radius:8px; background:var(--orange);
    color:#fff; font-size:.95rem; font-weight:600; cursor:pointer; transition:all .2s;
  }}
  .login-box button:hover {{ background:#e8701a; transform: translateY(-1px); }}
  .error {{ color:var(--red); font-size:.9rem; margin-bottom:1rem; padding:.5rem; 
            background: #ffeae8; border-radius:6px; border:1px solid #ffc4c4; }}
</style>
</head>
<body>
<form class="login-box" method="POST">
  <div class="logo">
    <img src="/static/ekolan_logo.png" alt="Ekolan">
  </div>
  <h1>PZT Compare</h1>
  <p>Porównywanie planów zagospodarowania terenu</p>
  {"<div class='error'>" + error + "</div>" if error else ""}
  <input type="password" name="password" placeholder="Hasło" autofocus>
  <button type="submit">Wejdź</button>
</form>
</body>
</html>'''
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['COMPARISONS_FOLDER'] = 'comparisons'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

# Ensure folders exist
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)
Path(app.config['COMPARISONS_FOLDER']).mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def pdf_to_png(pdf_path, output_path, dpi=150):
    """Convert PDF to PNG using pdftoppm"""
    try:
        cmd = ['pdftoppm', '-png', '-r', str(dpi), '-singlefile', pdf_path, output_path.rsplit('.', 1)[0]]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return True
        else:
            print(f"pdftoppm error: {result.stderr}")
            return False
    except Exception as e:
        print(f"PDF conversion error: {e}")
        return False

def load_comparison_meta(comp_id):
    """Load comparison metadata"""
    meta_path = Path(app.config['COMPARISONS_FOLDER']) / comp_id / 'meta.json'
    if meta_path.exists():
        with open(meta_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def save_comparison_meta(comp_id, meta_data):
    """Save comparison metadata"""
    comp_dir = Path(app.config['COMPARISONS_FOLDER']) / comp_id
    comp_dir.mkdir(exist_ok=True)
    meta_path = comp_dir / 'meta.json'
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta_data, f, indent=2, ensure_ascii=False)

def list_comparisons():
    """List all available comparisons"""
    comparisons = []
    comp_dir = Path(app.config['COMPARISONS_FOLDER'])
    if comp_dir.exists():
        for item in comp_dir.iterdir():
            if item.is_dir():
                meta = load_comparison_meta(item.name)
                if meta:
                    comparisons.append({
                        'id': item.name,
                        'name': meta.get('name', 'Unnamed'),
                        'description': meta.get('description', ''),
                        'created_date': meta.get('created_date', ''),
                        'preview_v2': f"comparisons/{item.name}/{meta.get('original_v2', '')}"
                    })
    return sorted(comparisons, key=lambda x: x.get('created_date', ''), reverse=True)

def calculate_transformation(points1, points2, img1_size, img2_size):
    """Calculate transformation from calibration points (normalized 0-1 coords)"""
    # Convert normalized coords to pixel coords
    w1, h1 = img1_size
    w2, h2 = img2_size
    pts1 = np.array([[p[0]*w1, p[1]*h1] for p in points1], dtype=np.float32)
    pts2 = np.array([[p[0]*w2, p[1]*h2] for p in points2], dtype=np.float32)
    
    n = min(len(pts1), len(pts2))
    pts1 = pts1[:n]
    pts2 = pts2[:n]
    
    try:
        if n >= 4:
            matrix, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
        else:
            # 3 points → affine transform
            matrix = cv2.getAffineTransform(pts1[:3], pts2[:3])
            # Convert 2x3 to 3x3 for warpPerspective
            matrix = np.vstack([matrix, [0, 0, 1]])
        return matrix
    except Exception as e:
        print(f"Transformation error: {e}")
        return None

def apply_transformation(img, matrix, target_size):
    """Apply homography transformation to image array"""
    width, height = target_size
    transformed = cv2.warpPerspective(img, matrix, (width, height))
    return transformed

def create_tinted_layer(image_array, tint_color):
    """Create tinted layer from image array - tint lines, keep transparency for white"""
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_array.copy()
    
    h, w = gray.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    
    # White pixels → transparent
    # Near-white → semi-transparent tint
    # Dark pixels → opaque tint
    # Scale: alpha = 255 - gray (darker = more opaque)
    alpha = (255 - gray).astype(np.float32)
    
    # Threshold: very light pixels (>240) become fully transparent
    alpha[gray > 240] = 0
    
    # Boost contrast for mid-tones
    alpha = np.clip(alpha * 1.5, 0, 255).astype(np.uint8)
    
    rgba[:, :, 0] = tint_color[0]  # B
    rgba[:, :, 1] = tint_color[1]  # G
    rgba[:, :, 2] = tint_color[2]  # R
    rgba[:, :, 3] = alpha
    
    return rgba

def create_clean_layer(image_array):
    """Create clean layer with white->transparent, preserving original colors"""
    if len(image_array.shape) == 2:
        # Grayscale - convert to BGR
        image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
    
    h, w = image_array.shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    white_mask = gray > 240
    non_white_mask = ~white_mask
    
    # Preserve original colors (BGRA for cv2.imwrite)
    rgba[non_white_mask, 0] = image_array[non_white_mask, 0]  # B
    rgba[non_white_mask, 1] = image_array[non_white_mask, 1]  # G
    rgba[non_white_mask, 2] = image_array[non_white_mask, 2]  # R
    rgba[non_white_mask, 3] = 255
    rgba[white_mask, 3] = 0
    
    return rgba

def create_diff_layer(img1_array, img2_array):
    """Create difference layer"""
    if len(img1_array.shape) == 3:
        gray1 = cv2.cvtColor(img1_array, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = img1_array.copy()
        
    if len(img2_array.shape) == 3:
        gray2 = cv2.cvtColor(img2_array, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = img2_array.copy()
    
    # Calculate absolute difference
    diff = cv2.absdiff(gray1, gray2)
    
    # Threshold to get significant differences
    _, diff_thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    
    h, w = diff_thresh.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    
    # Red tint for differences
    diff_mask = diff_thresh > 0
    rgba[diff_mask] = [255, 80, 48, 255]  # Red
    
    return rgba

def apply_roi_crop(image_array, roi_data):
    """Apply ROI crop to image array
    roi_data: {x, y, width, height} in normalized coordinates 0-1
    """
    try:
        h, w = image_array.shape[:2]
        
        # Convert normalized coordinates to pixel coordinates
        x = int(roi_data['x'] * w)
        y = int(roi_data['y'] * h)
        crop_width = int(roi_data['width'] * w)
        crop_height = int(roi_data['height'] * h)
        
        # Ensure coordinates are within image bounds
        x = max(0, min(x, w))
        y = max(0, min(y, h))
        x2 = min(x + crop_width, w)
        y2 = min(y + crop_height, h)
        
        # Perform crop
        cropped = image_array[y:y2, x:x2]
        
        print(f"ROI crop: from {w}×{h} to {cropped.shape[1]}×{cropped.shape[0]}")
        return cropped
        
    except Exception as e:
        print(f"ROI crop error: {e}")
        return None

def generate_layers(comp_id, img1_path, img2_path, transformation_matrix=None, roi_data=None):
    """Generate all comparison layers"""
    comp_dir = Path(app.config['COMPARISONS_FOLDER']) / comp_id
    
    # Load images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        return False
    
    # Apply transformation FIRST (align images before any cropping)
    if transformation_matrix is not None:
        h2, w2 = img2.shape[:2]
        img1 = apply_transformation(img1, transformation_matrix, (w2, h2))
        if img1 is None:
            return False
    else:
        # No transformation — resize img1 to match img2
        h2, w2 = img2.shape[:2]
        h1, w1 = img1.shape[:2]
        if (h1, w1) != (h2, w2):
            img1 = cv2.resize(img1, (w2, h2), interpolation=cv2.INTER_LANCZOS4)
    
    # Apply ROI cropping AFTER alignment (so same region = same content on both)
    if roi_data is not None:
        print(f"Applying ROI crop (post-alignment): {roi_data}")
        img1 = apply_roi_crop(img1, roi_data)
        img2 = apply_roi_crop(img2, roi_data)
        if img1 is None or img2 is None:
            print("ROI crop failed")
            return False
    
    # Generate layers
    layers = {}
    
    # Blue tinted layer (v1)
    blue_layer = create_tinted_layer(img1, [68, 136, 255])  # Blue
    layers['v1_blue'] = comp_dir / 'layer_v1_blue.png'
    cv2.imwrite(str(layers['v1_blue']), blue_layer)
    
    # Red tinted layer (v2)
    red_layer = create_tinted_layer(img2, [238, 68, 51])  # Red
    layers['v2_red'] = comp_dir / 'layer_v2_red.png'
    cv2.imwrite(str(layers['v2_red']), red_layer)
    
    # Clean layers
    clean1_layer = create_clean_layer(img1)
    layers['v1_clean'] = comp_dir / 'layer_v1_clean.png'
    cv2.imwrite(str(layers['v1_clean']), clean1_layer)
    
    clean2_layer = create_clean_layer(img2)
    layers['v2_clean'] = comp_dir / 'layer_v2_clean.png'
    cv2.imwrite(str(layers['v2_clean']), clean2_layer)
    
    # Diff layer
    diff_layer = create_diff_layer(img1, img2)
    layers['diff'] = comp_dir / 'layer_diff.png'
    cv2.imwrite(str(layers['diff']), diff_layer)
    
    return True

def encode_image_to_base64(image_path):
    """Encode image to base64 for OpenAI Vision API"""
    try:
        with open(image_path, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

def analyze_differences(comp_id, img1_path, img2_path):
    """Analyze differences between two images and generate change markers.
    
    Uses morphological operations to find distinct change clusters,
    then optionally enriches with AI descriptions.
    """
    import sys
    log = lambda msg: print(f"[analyze] {msg}", file=sys.stderr, flush=True)
    
    try:
        comp_dir = Path(app.config['COMPARISONS_FOLDER']) / comp_id
        diff_layer_path = comp_dir / 'layer_diff.png'
        
        if not diff_layer_path.exists():
            log("Diff layer not found")
            return []
        
        # 1. Load diff layer
        diff_img = cv2.imread(str(diff_layer_path), cv2.IMREAD_UNCHANGED)
        if diff_img is None:
            log("Could not load diff layer")
            return []
        
        # Get binary mask of differences
        if len(diff_img.shape) == 3 and diff_img.shape[2] == 4:
            binary = diff_img[:, :, 3]  # Alpha channel for RGBA
        elif len(diff_img.shape) == 3:
            binary = cv2.cvtColor(diff_img, cv2.COLOR_BGR2GRAY)
        else:
            binary = diff_img.copy()
        
        _, binary = cv2.threshold(binary, 10, 255, cv2.THRESH_BINARY)
        
        img_h, img_w = binary.shape
        total_pixels = img_h * img_w
        diff_pixels = cv2.countNonZero(binary)
        diff_pct = diff_pixels / total_pixels * 100
        log(f"Diff: {diff_pixels}/{total_pixels} pixels ({diff_pct:.1f}%)")
        
        if diff_pixels == 0:
            log("No differences found")
            return []
        
        # 2. Use morphological close to merge nearby diff pixels into clusters,
        #    then dilate/erode to separate distinct regions
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
        
        # If diff is very large (>15% of image), use erosion to break into smaller regions
        if diff_pct > 15:
            log(f"Large diff ({diff_pct:.1f}%), applying erosion to split regions")
            kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
            closed = cv2.erode(closed, kernel_erode, iterations=2)
            # Re-dilate slightly to keep region centers accurate
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
            closed = cv2.dilate(closed, kernel_dilate, iterations=1)
        
        # 3. Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter: min area = 0.1% of image, max area = 50% of image
        min_area = max(500, total_pixels * 0.001)
        max_area = total_pixels * 0.5
        
        regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area <= area <= max_area:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    x, y, w, h = cv2.boundingRect(contour)
                    regions.append({
                        'cx': cx / img_w,
                        'cy': cy / img_h,
                        'x': x / img_w,
                        'y': y / img_h,
                        'w': w / img_w,
                        'h': h / img_h,
                        'area': area,
                        'area_pct': area / total_pixels * 100
                    })
        
        # Sort by position (top-left to bottom-right)
        regions.sort(key=lambda r: (r['cy'], r['cx']))
        
        log(f"Found {len(regions)} change regions (from {len(contours)} contours)")
        
        if not regions:
            # Fallback: if morphology killed everything, create one marker at center of diff
            log("No regions after filtering, creating single center marker")
            regions = [{'cx': 0.5, 'cy': 0.5, 'area': diff_pixels, 'area_pct': diff_pct,
                       'x': 0, 'y': 0, 'w': 1, 'h': 1}]
        
        # 4. Generate markers (CV-only first, then try AI enrichment)
        markers = []
        for i, region in enumerate(regions):
            markers.append({
                'id': i,
                'x': round(region['cx'] * 100, 1),
                'y': round(region['cy'] * 100, 1),
                'title': f'Zmiana {i+1}',
                'type': 'change',
                'old': '',
                'new': '',
                'note': f'Obszar: {region["area_pct"]:.1f}% rysunku',
                'auto_generated': True,
                'category': 'change',
                'severity': 'important'
            })
        
        log(f"Generated {len(markers)} CV markers")
        
        # 5. Try AI enrichment (optional, non-blocking)
        openai_api_key = os.environ.get('OPENAI_API_KEY')
        if openai_api_key and len(markers) > 0:
            log("Attempting AI enrichment...")
            try:
                ai_markers = _enrich_markers_with_ai(
                    openai_api_key, img1_path, img2_path, regions, markers
                )
                if ai_markers:
                    log(f"AI enrichment successful: {len(ai_markers)} markers")
                    return ai_markers
                else:
                    log("AI enrichment returned empty, using CV markers")
            except Exception as e:
                log(f"AI enrichment failed: {e}, using CV markers")
        
        return markers
        
    except Exception as e:
        import traceback
        print(f"[analyze] FATAL: {e}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        return []


def _enrich_markers_with_ai(api_key, img1_path, img2_path, regions, cv_markers):
    """Try to add AI descriptions to CV-detected markers. Returns enriched markers or None."""
    img1_b64 = encode_image_to_base64(img1_path)
    img2_b64 = encode_image_to_base64(img2_path)
    
    if not img1_b64 or not img2_b64:
        return None
    
    region_desc = "\n".join(
        f"Zmiana {i+1}: pozycja ({r['cx']:.0%}, {r['cy']:.0%}), "
        f"bounding box ({r['x']:.0%},{r['y']:.0%}) rozmiar ({r['w']:.0%}×{r['h']:.0%}), "
        f"obszar {r['area_pct']:.1f}% rysunku"
        for i, r in enumerate(regions)
    )
    
    prompt = f"""Porównaj te dwa rysunki architektoniczne (PZT - Projekt Zagospodarowania Terenu).
Wykryto {len(regions)} regionów różnic:

{region_desc}

Dla KAŻDEGO regionu podaj krótki opis zmiany i przypisz KATEGORIĘ z jedną z poniższych opcji:

KRYTYCZNE (critical):
- lot_division: Podział/scalenie działki  
- building_layout: Zmiana układu budynków (nowy, usunięty, przesunięty)
- land_use: Zmiana przeznaczenia terenu
- building_line: Zmiana linii zabudowy

ISTOTNE (important):
- building_dimensions: Zmiana wymiarów budynku / pow. zabudowy
- bio_area: Zmiana powierzchni biologicznie czynnej
- infrastructure: Infrastruktura (przyłącza, sieci, zjazdy)
- setbacks: Strefy/odległości od granic

DROBNE (minor):
- dimensions: Wymiary/koty na rysunku
- geodetic: Dane geodezyjne (numery działek, rzędne)
- legend: Legenda/opis/tabliczka tytułowa
- graphics: Kolorystyka/szrafury/oznaczenia

Odpowiedz TYLKO poprawnym JSON (bez markdown):
{{"changes": [
  {{"id": 0, "title": "krótki tytuł", "type": "new|change|deleted", "was": "co było", "is": "co jest teraz", "note": "opcjonalny komentarz", "category": "category_name", "severity": "critical|important|minor"}},
  ...
]}}"""

    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img1_b64}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img2_b64}"}}
            ]
        }],
        max_tokens=3000,
        timeout=120
    )
    
    ai_text = response.choices[0].message.content
    print(f"[analyze] AI raw response: {ai_text[:500]}", flush=True)
    
    # Parse JSON
    import re
    json_match = re.search(r'\{.*\}', ai_text, re.DOTALL)
    if not json_match:
        return None
    
    ai_data = json.loads(json_match.group())
    changes = ai_data.get('changes', ai_data.get('regions', []))
    
    if not changes:
        return None
    
    type_map = {'new': 'new', 'change': 'change', 'delete': 'deleted', 'deleted': 'deleted'}
    
    # Category to severity mapping for fallback
    category_severity_map = {
        'lot_division': 'critical', 'building_layout': 'critical',
        'land_use': 'critical', 'building_line': 'critical',
        'building_dimensions': 'important', 'bio_area': 'important',
        'infrastructure': 'important', 'setbacks': 'important',
        'dimensions': 'minor', 'geodetic': 'minor',
        'legend': 'minor', 'graphics': 'minor',
        'change': 'important'
    }
    
    enriched = []
    for i, marker in enumerate(cv_markers):
        m = dict(marker)
        if i < len(changes):
            c = changes[i]
            m['title'] = c.get('title', m['title'])
            m['type'] = type_map.get(c.get('type', 'change'), 'change')
            m['old'] = c.get('was', c.get('old', ''))
            m['new'] = c.get('is', c.get('new', ''))
            m['note'] = c.get('note', m['note'])
            
            # Add category and severity
            category = c.get('category', 'change')
            severity = c.get('severity')
            
            # Fallback: map category to severity if severity not provided
            if not severity:
                severity = category_severity_map.get(category, 'important')
            
            m['category'] = category
            m['severity'] = severity
        else:
            # CV-only marker defaults
            m['category'] = 'change'
            m['severity'] = 'important'
            
        enriched.append(m)
    
    return enriched

def snap_to_feature(image_path, x_percent, y_percent, region_size=50, snap_radius=30):
    """Snap click point to nearest detected feature (corner/intersection)"""
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return x_percent, y_percent, False
        
        h, w = img.shape[:2]
        
        # Convert percentages to pixel coordinates
        x_px = int(x_percent * w / 100)
        y_px = int(y_percent * h / 100)
        
        # Define region around click
        x1 = max(0, x_px - region_size)
        y1 = max(0, y_px - region_size)
        x2 = min(w, x_px + region_size)
        y2 = min(h, y_px + region_size)
        
        # Extract region
        region = img[y1:y2, x1:x2]
        if region.size == 0:
            return x_percent, y_percent, False
        
        # Convert to grayscale
        gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        # Detect corners/features
        corners = cv2.goodFeaturesToTrack(
            gray_region,
            maxCorners=20,
            qualityLevel=0.01,
            minDistance=5,
            blockSize=3,
            useHarrisDetector=True,
            k=0.04
        )
        
        if corners is None or len(corners) == 0:
            return x_percent, y_percent, False
        
        # Find closest feature to click point within region
        click_x_region = x_px - x1
        click_y_region = y_px - y1
        
        min_dist = float('inf')
        best_corner = None
        
        for corner in corners:
            corner_x, corner_y = corner.ravel()
            dist = np.sqrt((corner_x - click_x_region)**2 + (corner_y - click_y_region)**2)
            
            if dist < min_dist and dist <= snap_radius:
                min_dist = dist
                best_corner = (corner_x, corner_y)
        
        if best_corner is None:
            return x_percent, y_percent, False
        
        # Convert back to image coordinates and percentages
        snapped_x_px = x1 + best_corner[0]
        snapped_y_px = y1 + best_corner[1]
        
        snapped_x_percent = (snapped_x_px / w) * 100
        snapped_y_percent = (snapped_y_px / h) * 100
        
        return snapped_x_percent, snapped_y_percent, True
        
    except Exception as e:
        print(f"Feature snapping error: {e}")
        return x_percent, y_percent, False

@app.route('/snap_point', methods=['POST'])
def snap_point():
    """Snap calibration point to nearest detected feature"""
    try:
        data = request.json
        session_id = data.get('session_id')
        x = data.get('x')  # percentage
        y = data.get('y')  # percentage
        version = data.get('version')  # 1 or 2
        
        if not session_id or x is None or y is None or version is None:
            return jsonify({'error': 'Missing parameters'}), 400
        
        session_dir = Path(app.config['UPLOAD_FOLDER']) / session_id
        if not session_dir.exists():
            return jsonify({'error': 'Session not found'}), 404
        
        # Get image path
        image_path = session_dir / f'v{version}.png'
        if not image_path.exists():
            return jsonify({'error': 'Image not found'}), 404
        
        # Perform feature snapping
        snapped_x, snapped_y, snapped = snap_to_feature(str(image_path), x, y)
        
        return jsonify({
            'x': snapped_x,
            'y': snapped_y,
            'snapped': snapped,
            'original_x': x,
            'original_y': y
        })
        
    except Exception as e:
        print(f"Snap point error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    """Main page - list comparisons"""
    comparisons = list_comparisons()
    return render_template('index.html', comparisons=comparisons)

@app.route('/new')
def new_comparison():
    """New comparison upload form"""
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file upload"""
    if 'file1' not in request.files or 'file2' not in request.files:
        return jsonify({'error': 'Both files required'}), 400
    
    file1 = request.files['file1']
    file2 = request.files['file2']
    
    if not (file1 and allowed_file(file1.filename) and file2 and allowed_file(file2.filename)):
        return jsonify({'error': 'Invalid file format'}), 400
    
    # Generate session ID for calibration
    session_id = str(uuid.uuid4())
    session_dir = Path(app.config['UPLOAD_FOLDER']) / session_id
    session_dir.mkdir(exist_ok=True)
    
    # Save uploaded files
    file1_path = session_dir / f"v1.{file1.filename.rsplit('.', 1)[1].lower()}"
    file2_path = session_dir / f"v2.{file2.filename.rsplit('.', 1)[1].lower()}"
    
    file1.save(str(file1_path))
    file2.save(str(file2_path))
    
    # Convert PDFs to PNG if needed
    png1_path = session_dir / "v1.png"
    png2_path = session_dir / "v2.png"
    
    if file1_path.suffix.lower() == '.pdf':
        if not pdf_to_png(str(file1_path), str(png1_path)):
            return jsonify({'error': 'PDF conversion failed for file 1'}), 500
    else:
        # Convert to PNG for consistency
        img = Image.open(file1_path)
        img.convert('RGB').save(png1_path)
    
    if file2_path.suffix.lower() == '.pdf':
        if not pdf_to_png(str(file2_path), str(png2_path)):
            return jsonify({'error': 'PDF conversion failed for file 2'}), 500
    else:
        img = Image.open(file2_path)
        img.convert('RGB').save(png2_path)
    
    return jsonify({
        'session_id': session_id,
        'preview1': f"/uploads/{session_id}/v1.png",
        'preview2': f"/uploads/{session_id}/v2.png"
    })

@app.route('/calibrate', methods=['POST'])
def calibrate():
    """Process calibration points and generate comparison"""
    try:
        data = request.json
        session_id = data.get('session_id')
        points1 = data.get('points1', [])
        points2 = data.get('points2', [])
        name = data.get('name', 'Unnamed Comparison')
        description = data.get('description', '')
        v1_name = data.get('v1_name', 'Version 1')
        v2_name = data.get('v2_name', 'Version 2')
        roi_data = data.get('roi')  # ROI data {x, y, width, height} in 0-1 range or None
        
        if not session_id or len(points1) < 3 or len(points2) < 3:
            return jsonify({'error': 'Insufficient calibration points'}), 400
        
        session_dir = Path(app.config['UPLOAD_FOLDER']) / session_id
        if not session_dir.exists():
            return jsonify({'error': 'Session not found'}), 404
        
        # Get image dimensions for coordinate conversion
        img1_cv = cv2.imread(str(session_dir / 'v1.png'))
        img2_cv = cv2.imread(str(session_dir / 'v2.png'))
        if img1_cv is None or img2_cv is None:
            return jsonify({'error': 'Could not read uploaded images'}), 500
        
        img1_size = (img1_cv.shape[1], img1_cv.shape[0])  # (w, h)
        img2_size = (img2_cv.shape[1], img2_cv.shape[0])
        
        # Calculate transformation
        transformation = calculate_transformation(points1, points2, img1_size, img2_size)
        
        # Generate comparison ID
        comp_id = str(uuid.uuid4())
        comp_dir = Path(app.config['COMPARISONS_FOLDER']) / comp_id
        comp_dir.mkdir(exist_ok=True)
        
        # Copy original images
        orig1_path = comp_dir / 'original_v1.png'
        orig2_path = comp_dir / 'original_v2.png'
        
        import shutil
        shutil.copy(session_dir / 'v1.png', orig1_path)
        shutil.copy(session_dir / 'v2.png', orig2_path)
        
        # Generate layers
        success = generate_layers(comp_id, str(orig1_path), str(orig2_path), transformation, roi_data)
        if not success:
            return jsonify({'error': 'Layer generation failed'}), 500
        
        # AI Analysis - automatically find and describe differences
        import sys, traceback as tb
        print(f"[DEBUG] Starting analyze_differences for {comp_id}", file=sys.stderr, flush=True)
        try:
            ai_markers = analyze_differences(comp_id, str(orig1_path), str(orig2_path))
            print(f"[DEBUG] analyze_differences returned {len(ai_markers)} markers", file=sys.stderr, flush=True)
        except Exception as analyze_err:
            print(f"[DEBUG] analyze_differences CRASHED: {analyze_err}", file=sys.stderr, flush=True)
            tb.print_exc(file=sys.stderr)
            ai_markers = []
        
        # Save metadata
        meta_data = {
            'id': comp_id,
            'name': name,
            'description': description,
            'created_date': datetime.now().isoformat(),
            'v1_name': v1_name,
            'v2_name': v2_name,
            'original_v1': 'original_v1.png',
            'original_v2': 'original_v2.png',
            'roi': roi_data,  # Store ROI data
            'layers': {
                'v1_blue': 'layer_v1_blue.png',
                'v2_red': 'layer_v2_red.png',
                'v1_clean': 'layer_v1_clean.png',
                'v2_clean': 'layer_v2_clean.png',
                'diff': 'layer_diff.png'
            },
            'calibration_points': {
                'points1': points1,
                'points2': points2
            },
            'markers': ai_markers
        }
        
        save_comparison_meta(comp_id, meta_data)
        
        # Clean up session
        shutil.rmtree(session_dir, ignore_errors=True)
        
        return jsonify({'comparison_id': comp_id})
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/compare/<comp_id>')
def view_comparison(comp_id):
    """View comparison interface"""
    meta = load_comparison_meta(comp_id)
    if not meta:
        return "Comparison not found", 404
    
    return render_template('compare.html', comparison=meta)

@app.route('/api/markers/<comp_id>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def manage_markers(comp_id):
    """Manage comparison markers"""
    meta = load_comparison_meta(comp_id)
    if not meta:
        return jsonify({'error': 'Comparison not found'}), 404
    
    if request.method == 'GET':
        return jsonify(meta.get('markers', []))
    
    elif request.method == 'POST':
        # Add new marker
        marker_data = request.json
        marker_data['id'] = len(meta.get('markers', []))
        meta.setdefault('markers', []).append(marker_data)
        save_comparison_meta(comp_id, meta)
        return jsonify(marker_data)
    
    elif request.method == 'PUT':
        # Update marker
        marker_id = request.json.get('id')
        markers = meta.get('markers', [])
        for i, marker in enumerate(markers):
            if marker.get('id') == marker_id:
                markers[i] = request.json
                save_comparison_meta(comp_id, meta)
                return jsonify(markers[i])
        return jsonify({'error': 'Marker not found'}), 404
    
    elif request.method == 'DELETE':
        # Delete marker
        marker_id = request.args.get('id', type=int)
        markers = meta.get('markers', [])
        meta['markers'] = [m for m in markers if m.get('id') != marker_id]
        save_comparison_meta(comp_id, meta)
        return jsonify({'success': True})

@app.route('/api/delete/<comp_id>', methods=['DELETE'])
def delete_comparison(comp_id):
    """Delete comparison"""
    comp_dir = Path(app.config['COMPARISONS_FOLDER']) / comp_id
    if comp_dir.exists():
        import shutil
        shutil.rmtree(comp_dir)
        return jsonify({'success': True})
    return jsonify({'error': 'Comparison not found'}), 404

# Static file serving
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route('/uploads/<path:filename>')
def serve_uploads(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/comparisons/<path:filename>')
def serve_comparisons(filename):
    return send_from_directory(app.config['COMPARISONS_FOLDER'], filename)

@app.route('/admin/purge-all', methods=['POST'])
@requires_auth
def purge_all_comparisons():
    """Delete ALL comparisons from the volume"""
    import shutil
    comp_dir = Path(app.config['COMPARISONS_FOLDER'])
    deleted = []
    if comp_dir.exists():
        for item in comp_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
                deleted.append(item.name)
    return jsonify({'deleted': deleted, 'count': len(deleted)})

@app.route('/health')
def health():
    return 'ok'

@app.route('/debug/reanalyze/<comp_id>')
@requires_auth
def debug_reanalyze(comp_id):
    """Debug: re-run analyze_differences on existing comparison"""
    import sys, traceback as tb
    meta = load_comparison_meta(comp_id)
    if not meta:
        return jsonify({'error': 'Not found'}), 404
    
    comp_dir = Path(app.config['COMPARISONS_FOLDER']) / comp_id
    orig1 = comp_dir / 'original_v1.png'
    orig2 = comp_dir / 'original_v2.png'
    diff_path = comp_dir / 'layer_diff.png'
    
    debug_info = {
        'comp_id': comp_id,
        'orig1_exists': orig1.exists(),
        'orig2_exists': orig2.exists(),
        'diff_exists': diff_path.exists(),
        'openai_key_set': bool(os.environ.get('OPENAI_API_KEY')),
    }
    
    if diff_path.exists():
        import cv2
        diff_img = cv2.imread(str(diff_path), cv2.IMREAD_UNCHANGED)
        if diff_img is not None:
            debug_info['diff_shape'] = list(diff_img.shape)
            debug_info['diff_channels'] = diff_img.shape[2] if len(diff_img.shape) == 3 else 1
            
            # Check alpha
            if len(diff_img.shape) == 3 and diff_img.shape[2] == 4:
                alpha = diff_img[:,:,3]
            else:
                alpha = cv2.cvtColor(diff_img, cv2.COLOR_BGR2GRAY) if len(diff_img.shape) == 3 else diff_img
            
            nonzero = int(cv2.countNonZero(alpha))
            total = alpha.shape[0] * alpha.shape[1]
            debug_info['nonzero_pixels'] = nonzero
            debug_info['total_pixels'] = total
            debug_info['diff_percent'] = round(nonzero / total * 100, 1)
            
            contours, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            debug_info['total_contours'] = len(contours)
            significant = [c for c in contours if cv2.contourArea(c) >= 500]
            debug_info['significant_contours'] = len(significant)
    
    try:
        markers = analyze_differences(comp_id, str(orig1), str(orig2))
        debug_info['markers_count'] = len(markers)
        debug_info['markers'] = markers
    except Exception as e:
        debug_info['error'] = str(e)
        debug_info['traceback'] = tb.format_exc()
    
    return jsonify(debug_info)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8899))
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    print(f"🏗️ PZT Compare v2 starting on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=debug)