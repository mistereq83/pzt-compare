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

# --- HTTP Basic Auth ---
AUTH_USER = os.environ.get('PZT_USER', 'molab')
AUTH_PASS = os.environ.get('PZT_PASS', 'pzt2026!')

def check_auth(username, password):
    return username == AUTH_USER and password == AUTH_PASS

def authenticate():
    return Response('Wymagane logowanie.', 401, {'WWW-Authenticate': 'Basic realm="PZT Compare"'})

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated

@app.before_request
def before_request_auth():
    # Skip auth for health check
    if request.path == '/health':
        return
    auth = request.authorization
    if not auth or not check_auth(auth.username, auth.password):
        return authenticate()
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
    
    # Apply ROI cropping if specified (BEFORE transformation)
    if roi_data is not None:
        print(f"Applying ROI crop: {roi_data}")
        img1 = apply_roi_crop(img1, roi_data)
        img2 = apply_roi_crop(img2, roi_data)
        if img1 is None or img2 is None:
            print("ROI crop failed")
            return False
    
    # Apply transformation if provided
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
    """Analyze differences using AI vision and generate markers automatically"""
    try:
        comp_dir = Path(app.config['COMPARISONS_FOLDER']) / comp_id
        diff_layer_path = comp_dir / 'layer_diff.png'
        
        if not diff_layer_path.exists():
            print("Diff layer not found")
            return []
        
        # 1. Load diff layer and find change regions
        diff_img = cv2.imread(str(diff_layer_path), cv2.IMREAD_UNCHANGED)
        if diff_img is None:
            print("Could not load diff layer")
            return []
        
        # Convert to grayscale for contour detection
        if len(diff_img.shape) == 4:
            gray_diff = diff_img[:, :, 3]  # Alpha channel
        else:
            gray_diff = cv2.cvtColor(diff_img, cv2.COLOR_BGR2GRAY)
        
        # Find contours of change regions
        contours, _ = cv2.findContours(gray_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter significant regions (area > threshold)
        min_area = 500  # Minimum area in pixels
        significant_regions = []
        
        img_height, img_width = gray_diff.shape
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                # Calculate centroid
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Convert to normalized coordinates (0-1)
                    norm_x = cx / img_width
                    norm_y = cy / img_height
                    
                    significant_regions.append({
                        'x': norm_x,
                        'y': norm_y,
                        'area': area
                    })
        
        if not significant_regions:
            print("No significant change regions found")
            return []
        
        print(f"Found {len(significant_regions)} change regions")
        
        # 2. Prepare OpenAI Vision API request (optional — falls back to CV-only markers)
        openai_api_key = os.environ.get('OPENAI_API_KEY')
        if not openai_api_key:
            print("OPENAI_API_KEY not set — generating CV-only markers without AI descriptions")
            # Generate basic markers from contour detection alone
            cv_markers = []
            for i, region in enumerate(significant_regions):
                cv_markers.append({
                    'id': i,
                    'x': round(region['x'] * 100, 1),
                    'y': round(region['y'] * 100, 1),
                    'title': f'Zmiana {i+1}',
                    'type': 'change',
                    'old': 'Wykryto różnicę (brak opisu AI)',
                    'new': 'Wykryto różnicę (brak opisu AI)',
                    'note': f'Obszar zmiany: {region["area"]:.0f} px²',
                    'auto_generated': True
                })
            print(f"Generated {len(cv_markers)} CV-only markers")
            return cv_markers
        
        # Encode images to base64
        img1_b64 = encode_image_to_base64(img1_path)
        img2_b64 = encode_image_to_base64(img2_path)
        
        if not img1_b64 or not img2_b64:
            print("Failed to encode images to base64")
            return []
        
        # Prepare region positions for the prompt
        region_descriptions = []
        for i, region in enumerate(significant_regions):
            region_descriptions.append(f"Region {i+1}: pozycja {region['x']:.2f}, {region['y']:.2f} (x,y jako procent obrazu)")
        
        regions_text = "\n".join(region_descriptions)
        
        # 3. Send to OpenAI Vision API
        client = openai.OpenAI(api_key=openai_api_key)
        
        prompt = f"""Porównaj te dwa rysunki architektoniczne/budowlane. 
Na drugim rysunku zidentyfikowano {len(significant_regions)} regionów zmian w następujących pozycjach:

{regions_text}

Dla każdego regionu opisz:
- co BYŁO w wersji 1 (pierwszy rysunek)  
- co JEST w wersji 2 (drugi rysunek)
- jaki typ zmiany to jest (nowe/zmiana/usunięte)

Odpowiedz w formacie JSON:
{{
  "regions": [
    {{
      "region_id": 1,
      "title": "Krótki tytuł zmiany",
      "type": "new|change|delete", 
      "was": "Opis tego co było w wersji 1",
      "is": "Opis tego co jest w wersji 2",
      "note": "Dodatkowe szczegóły"
    }}
  ]
}}"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img1_b64}"
                                }
                            },
                            {
                                "type": "image_url", 
                                "image_url": {
                                    "url": f"data:image/png;base64,{img2_b64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=2000,
                timeout=30
            )
            
            ai_response = response.choices[0].message.content
            print(f"AI response: {ai_response}")
            
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return []
        
        # 4. Parse AI response and generate markers
        try:
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
            if json_match:
                ai_data = json.loads(json_match.group())
                regions_data = ai_data.get('regions', [])
            else:
                print("Could not find JSON in AI response")
                return []
            
            markers = []
            for i, region_data in enumerate(regions_data):
                if i < len(significant_regions):
                    # Map AI response to our region
                    region = significant_regions[i]
                    
                    # Determine badge based on type
                    change_type = region_data.get('type', 'change').lower()
                    badge_map = {
                        'new': 'new',
                        'change': 'chg', 
                        'delete': 'del'
                    }
                    badge = badge_map.get(change_type, 'chg')
                    
                    # Frontend expects x/y as percentage (0-100), type as full word
                    type_map = {'new': 'new', 'chg': 'change', 'del': 'deleted',
                                'change': 'change', 'delete': 'deleted'}
                    marker = {
                        'id': i,
                        'x': round(region['x'] * 100, 1),
                        'y': round(region['y'] * 100, 1),
                        'title': region_data.get('title', f'Zmiana {i+1}'),
                        'type': type_map.get(change_type, 'change'),
                        'old': region_data.get('was', ''),
                        'new': region_data.get('is', ''),
                        'note': region_data.get('note', ''),
                        'auto_generated': True
                    }
                    markers.append(marker)
            
            print(f"Generated {len(markers)} AI markers")
            return markers
            
        except json.JSONDecodeError as e:
            print(f"Failed to parse AI response JSON: {e}")
            return []
        
    except Exception as e:
        print(f"Error in analyze_differences: {e}")
        import traceback
        traceback.print_exc()
        return []

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
        ai_markers = analyze_differences(comp_id, str(orig1_path), str(orig2_path))
        
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
@app.route('/uploads/<path:filename>')
def serve_uploads(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/comparisons/<path:filename>')
def serve_comparisons(filename):
    return send_from_directory(app.config['COMPARISONS_FOLDER'], filename)

@app.route('/health')
def health():
    return 'ok'

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8899))
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    print(f"🏗️ PZT Compare v2 starting on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=debug)