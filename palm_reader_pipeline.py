import os
import boto3
import base64
import json
import tempfile
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import cv2
import numpy as np
import openai
import shutil
from pathlib import Path
from tenacity import retry, wait_exponential

# Configuration
S3_BUCKET = "palm-reader"
MODEL_S3_PATH = "s3://palm-reader/model-weights/best5.pt"
OPENAI_API_KEY = ""
PALM_CLASSES = {0: 'Fate-Line', 1: 'Head-Line', 2: 'Heart-Line', 3: 'Life-Line'}

# Initialize clients
s3 = boto3.client('s3')
openai.api_key = OPENAI_API_KEY

def process_palm_image(s3_image_path):
    """Main processing pipeline"""
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            # Download input image from S3
            img_path = download_from_s3(s3_image_path, tmpdir)
            
            # Run YOLO inference
            yolo_img_path, yolo_json_path = run_yolo_inference(img_path, tmpdir)
            
            # Process with GPT-4 Vision
            vision_json_path = process_with_vision(yolo_img_path, yolo_json_path, tmpdir)
            
            # Generate summary
            summary_json_path = generate_summary(vision_json_path, tmpdir)
            
            return {
                'output_image': yolo_img_path,
                'yolo_json': yolo_json_path,
                'vision_json': vision_json_path,
                'summary_json': summary_json_path
            }
        except Exception as e:
            print(f"Pipeline failed: {str(e)}")
            raise

def download_from_s3(s3_uri, dest_dir):
    """Improved S3 download with error handling"""
    try:
        bucket, key = parse_s3_uri(s3_uri)
        local_path = os.path.join(dest_dir, os.path.basename(key))
        
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3.download_file(bucket, key, local_path)
        print(f"Downloaded {s3_uri} to {local_path}")
        return local_path
    except Exception as e:
        print(f"Download failed: {str(e)}")
        raise

def upload_to_s3(local_path, s3_uri):
    """Robust S3 upload with validation"""
    try:
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Missing file: {local_path}")
            
        bucket, key = parse_s3_uri(s3_uri)
        s3.upload_file(local_path, bucket, key)
        print(f"Uploaded {local_path} to s3://{bucket}/{key}")
        return s3_uri
    except Exception as e:
        print(f"Upload failed: {str(e)}")
        raise

def parse_s3_uri(s3_uri):
    """Improved S3 URI parsing"""
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    parts = s3_uri[5:].split("/", 1)
    return parts[0], parts[1] if len(parts) > 1 else ""

def run_yolo_inference(img_path, tmpdir):
    """Fixed YOLO inference flow"""
    try:
        # Download model
        model_path = download_from_s3(MODEL_S3_PATH, tmpdir)
        
        # Run inference
        model = YOLO(model_path)
        results = model(img_path, retina_masks=True)
        r = results[0]
        
        # Generate output paths
        base_name = os.path.basename(img_path).split('.')[0]
        output_img_name = f"{base_name}_result.jpg"
        output_img_path = os.path.join(tmpdir, output_img_name)
        json_name = f"{base_name}_yolo.json"
        json_path = os.path.join(tmpdir, json_name)

        
        # Process and save outputs
        process_yolo_output(r, output_img_path)
        generate_yolo_json(r, json_path)
        
        # Upload results
        output_s3 = f"s3://{S3_BUCKET}/results/{output_img_name}"
        json_s3 = f"s3://{S3_BUCKET}/results/{json_name}"
        
        upload_to_s3(output_img_path, output_s3)
        upload_to_s3(json_path, json_s3)
        
        return output_s3, json_s3
    except Exception as e:
        print(f"YOLO inference failed: {str(e)}")
        raise

def process_yolo_output(r, output_path):
    """Complete YOLO output processing"""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Actual YOLO processing code from notebook
        PIXEL_SHRINK = 5
        MASK_ALPHA = 0.5
        COLOR_SEED = 42
        LABEL_MARGIN_PX = 2
        FONT_RATIO = 0.005
        FONT_MIN = 8
        FONT_MAX = 32
        BOX_THICK_RATIO = 0.005
        MASK_THICK_RATIO = 0.0025
        THICK_MIN = 1
        THICK_MAX = 20
        FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
        TEXT_THICKNESS = 3

        img = r.orig_img.copy()
        H, W = img.shape[:2]

        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)
        scores = r.boxes.conf.cpu().numpy()
        names = r.names
        masks = r.masks.data.cpu().numpy() if r.masks is not None else []

        rng = np.random.default_rng(COLOR_SEED)
        colors = [tuple(int(c) for c in rng.integers(0, 256, 3)) for _ in range(len(boxes))]

        eroded_masks = []
        if len(masks) > 0:
            kernel = np.ones((2 * PIXEL_SHRINK + 1, 2 * PIXEL_SHRINK + 1), np.uint8)
            for m in masks:
                bm = (m * 255).astype(np.uint8)
                ed = cv2.erode(bm, kernel, iterations=1)
                eroded_masks.append((ed > 0).astype(np.uint8))
            eroded_masks = np.stack(eroded_masks, axis=0)
            composite_mask = np.any(eroded_masks, axis=0)
        else:
            composite_mask = np.zeros((H, W), dtype=bool)

        def shrink_box(b, d):
            x1, y1, x2, y2 = b
            return [x1 + d, y1 + d, x2 - d, y2 - d]

        shrunk_boxes = np.array([shrink_box(b, PIXEL_SHRINK) for b in boxes], dtype=int)

        font_size = int(np.clip(H * FONT_RATIO, FONT_MIN, FONT_MAX))
        box_thick = int(np.clip(H * BOX_THICK_RATIO, THICK_MIN, THICK_MAX))
        mask_thick = int(np.clip(H * MASK_THICK_RATIO, THICK_MIN, THICK_MAX))

        for idx, em in enumerate(eroded_masks):
            color = colors[idx]
            mask_bool = em.astype(bool)
            img[mask_bool] = (
                img[mask_bool] * (1 - MASK_ALPHA) + np.array(color) * MASK_ALPHA
            ).astype(np.uint8)
            cnts, _ = cv2.findContours((em * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cnts, -1, color, thickness=mask_thick, lineType=cv2.LINE_AA)

        annotator = Annotator(img, line_width=box_thick, font_size=font_size)
        label_rects = []

        for idx, (box, cls, conf) in enumerate(zip(shrunk_boxes, classes, scores)):
            annotator.box_label(box.tolist(), '', color=colors[idx])
            label = f"{names[int(cls)]} {conf:.2f}"
            (txt_w, txt_h), baseline = cv2.getTextSize(label, FONT_FACE, font_size / 10, thickness=TEXT_THICKNESS)
            lx, ly = box[0], box[1] - LABEL_MARGIN_PX
            if ly - txt_h - baseline < 0:
                ly = box[3] + txt_h + LABEL_MARGIN_PX
            rect = [lx, ly - txt_h - baseline, lx + txt_w, ly]

            def overlaps_any(r, prev_rects):
                for p in prev_rects:
                    if not (r[2] < p[0] or r[0] > p[2] or r[3] < p[1] or r[1] > p[3]):
                        return True
                x0, y0, x1, y1 = r
                return np.any(composite_mask[y0:y1, x0:x1])

            while overlaps_any(rect, label_rects):
                rect[1] += txt_h + baseline + LABEL_MARGIN_PX
                rect[3] += txt_h + baseline + LABEL_MARGIN_PX

            label_rects.append(rect)

            cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3] + baseline), colors[idx], thickness=cv2.FILLED)
            cv2.putText(img, label, (rect[0], rect[3]), FONT_FACE, font_size / 10, (255, 255, 255), thickness=TEXT_THICKNESS, lineType=cv2.LINE_AA)

        img = annotator.result()

        cv2.imwrite(output_path, img)
        if not os.path.exists(output_path):
            raise RuntimeError("Failed to save YOLO output image")
            
        return output_path
    except Exception as e:
        print(f"YOLO processing failed: {str(e)}")
        raise

def generate_yolo_json(r, json_path):
    """Save YOLO results to JSON in notebook format"""
    try:
        # Get raw JSON data
        json_data = json.loads(r.to_json(normalize=False, decimals=5))
        
        # Transform to match notebook format
        minimal_list = [
            {
                "name": PALM_CLASSES[det["class"]],  # Use class mapping dictionary
                "class": det["class"],
                "confidence": det["confidence"],
                "box": det["box"]
            } for det in json_data
        ]
        
        with open(json_path, 'w') as f:
            json.dump(minimal_list, f, indent=4)
            
        return json_path
    except Exception as e:
        print(f"JSON generation failed: {str(e)}")
        raise

def process_with_vision(img_s3_path, json_s3_path, tmpdir):
    """Process with GPT-4 Vision"""
    # Download files
    img_path = download_from_s3(img_s3_path, tmpdir)
    json_path = download_from_s3(json_s3_path, tmpdir)
    
    # Encode image
    with open(img_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    # Load YOLO JSON
    with open(json_path) as f:
        yolo_data = json.load(f)
    
    # Call OpenAI
    vision_json = extract_palm_features(encoded_image, yolo_data)
    
    # Save and upload
    vision_path = os.path.join(tmpdir, "vision_output.json")
    with open(vision_path, 'w') as f:
        json.dump(vision_json, f)
    
    vision_s3_path = img_s3_path.replace(".jpg", "_vision.json")
    upload_to_s3(vision_path, vision_s3_path)
    return vision_s3_path

def generate_summary(vision_s3_path, tmpdir):
    """Generate summary with GPT-4"""
    # Download vision JSON
    vision_path = download_from_s3(vision_s3_path, tmpdir)
    
    with open(vision_path) as f:
        vision_data = json.load(f)
    
    # Call OpenAI
    summary = extract_palm_features_summaries(vision_data)
    
    # Save and upload
    summary_path = os.path.join(tmpdir, "summary_output.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f)
    
    summary_s3_path = vision_s3_path.replace("_vision.json", "_summary.json")
    upload_to_s3(summary_path, summary_s3_path)
    return summary_s3_path

@retry(wait=wait_exponential(multiplier=1, min=4, max=60))
def extract_palm_features(image_base64, yolo_json_data):
    """
    Analyzes a palm image and YOLOv11 JSON data to extract palm line features using GPT-4o.
    """
    if not image_base64:
        return "Error: Image data is missing."

    prompt = f"""
    You are an expert palmistry analyst. Analyze the hand image and YOLOv11 JSON data using these **metric-driven protocols**:
    
    **Operational Definitions**
    1. **Length**: (arc_length/max_palm_width)×100 → short(<40%), medium(40-75%), long(>75%)
    2. **Depth**: deep if stroke_width ≤1.5×background_crease, else shallow
    3. **Branches**: none/upward/downward/mixed (splits ≥15% length at <45°)
    4. **Forks**: true if >25% width separation at distal segment
    5. **Breaks**: true if discontinuity >5% palm width
    6. **Coordinates**: Normalized to [0-1] scale relative to palm bounding box
    
    JSON data:
    ```json
    {json.dumps(yolo_json_data, indent=2)}
    
    Required Features per Line
    
    length: short/medium/long/not_determinable (calculated via palm width %)
    
    depth: deep/shallow/not_determinable (stroke width ratio)
    
    clarity: clear/chained/blurry/not_determinable
    
    curvature: straight(<5°)/curved/downward/irregular/not_determinable
    
    branches: none/upward/downward/mixed (categorical)
    
    forks: true/false (major fork)
    
    breaks: true/false (significant discontinuity)
    
    islands: true/false (closed loops override forks)
    
    start_mount: Jupiter/Saturn/Apollo/Mercury/Venus/Moon/unknown
    
    end_mount: Jupiter/Saturn/Apollo/Mercury/Venus/Moon/unknown
    
    start_xy: [x,y] (normalized 0-1 coordinates)
    
    end_xy: [x,y] (normalized 0-1 coordinates)
    
    Output Format 
    
    {{
      "hand": "right/left/unknown",
      "lines": [
        {{
          "line_type": "heart-line", 
          "yolo_confidence": 0.90569,
          "yolo_box": {{ "x1": 609.73, "y1": 702.86, "x2": 911.74, "y2": 848.86 }},
          "features": {{
            "length": "medium",
            "depth": "deep",
            "clarity": "chained",
            "curvature": "straight",
            "branches": "upward",  
            "forks": false,
            "breaks": false,
            "islands": true,
            "start_mount": "Jupiter",
            "end_mount": "Mercury",
            "start_xy": [0.18, 0.12],
            "end_xy": [0.78, 0.14]
          }}
        }}
      ],
      "variant": "simian",  
      "quality_issues": ["low_confidence"]  // Confidence <0.6
    }}
    
    Analyze the image carefully. Use the bounding box information from the JSON to focus on the correct part of the image for each line. Be as detailed and accurate as possible. If a line mentioned in the JSON is not clearly visible or its features cannot be determined from the image, use "not determinable" for the descriptive string features. For 'branches', 'forks', 'breaks', and 'islands', if their presence cannot be determined due to poor visibility of the line segment, then outputting false is acceptable, but prioritize detection if the line segment is clear.
    """
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert palmistry analyst."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]}
            ],
            max_tokens=2000, 
            temperature=0.2 
        )
        if not response.choices or not response.choices[0].message or not response.choices[0].message.content:
            print("Warning: No content returned from GPT-4o.")
            return "Error: No content returned from GPT-4o."
        
        # The response content should be a JSON string, so we try to parse it
        response_content = response.choices[0].message.content.strip()
        
        # GPT might return the JSON block within triple backticks, remove them if present
        if response_content.startswith("```json"):
            response_content = response_content[7:]
        if response_content.endswith("```"):
            response_content = response_content[:-3]
            
        return json.loads(response_content) # Parse the JSON string into a Python dictionary

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response from GPT-4o: {e}")
        print(f"Raw response: {response_content}")
        return f"Error: Could not parse JSON response. Raw response: {response_content}"
    except Exception as e:
        print(f"An error occurred while calling OpenAI API: {e}")
        return f"Error: An API error occurred: {e}"

@retry(wait=wait_exponential(multiplier=1, min=4, max=60))
def extract_palm_features_summaries(vision_json_data):
    """
    Analyzes a palm JSON data to generate summary using GPT-4o.
    """
    
    prompt = f"""
    I have already processed the palm image and extracted its features into JSON below.  
    Please generate your interpretation **only** from that JSON—do not attempt any additional image analysis.  
    **Use only the pre‑extracted Attributes** (length, depth, clarity, curvature, branches, forks, breaks, islands, start_mount, end_mount) **from the JSON below**. Do **not** redefine or re‑extract them. For each of the lines, produce:
    1. An unchanged `"Attributes"` block populated exactly as in the JSON.  
    2. A `"Summary"`—a single, **positive**, **supportive**, **≤ 80‑word** narrative that uses *only* the cue‑to‑meaning phrases from the Interpretation Dictionary.
    
    Interpretation Dictionary (for summary generation only):
    
    **Head Line**  
    - Long: May indicate ambition.  
    - Short: Suggests intelligence and intuition.  
    - Deep: Reflects a strong memory.  
    - Straight: Indicates a practical and logical outlook.  
    - Broken: Represents evolving interests or adaptive thinking.  
    - Forked: Can signify versatile thinking or potential career change.  
    
    **Heart Line**  
    - Long: Reflects deep emotional intelligence and care.  
    - Ends below the index finger: Suggests idealism in relationships.  
    - Deep: Emphasizes the importance of close bonds.  
    - Broken: Indicates growth through relationships.  
    
    **Life Line**  
    - Long, curving: Shows energy, strength, and determination.  
    - Short, shallow: Suggests a reflective or adaptable nature.  
    - Straight, little curve: Indicates cautiousness and strong boundaries.  
    - Markings (islands, circles, crosses): Represent meaningful life events or turning points.  
    
    **Fate Line**  
    - Deep and long: Strong sense of purpose or destiny.  
    - Broken: Life changes that brought new opportunities.  
    - Starts joined with Life Line: Self‑made and independent.  
    - Wavy or chained: Dynamic path with varied experiences.  
    
    --- Extracted Palm Features (JSON Input) ---
    ```json
    {json.dumps(vision_json_data, indent=2)}
    
    Output Format (Model must not add, remove, or rename any keys; output only these fields.):
    
    {{
      "hand": "<from input JSON>",
          "Head_Line": {{
            "yolo_confidence": <from input JSON>,
            "yolo_box": {{
                "x1": <from input JSON>,
                "y1": <from input JSON>,
                "x2": <from input JSON>,
                "y2": <from input JSON>
            }},
            "Attributes": {{
              "length": "<from input JSON>",
              "depth": "<from input JSON>",
              "clarity": "<from input JSON>",
              "curvature": "<from input JSON>",
              "branches": "<from input JSON>",
              "forks": <from input JSON>,
              "breaks": <from input JSON>,
              "islands": <from input JSON>,
              "start_mount": "<from input JSON>",
              "end_mount": "<from input JSON>",
              "start_xy":    [<from input JSON>, <from input JSON>],
              "end_xy":      [<from input JSON>", <from input JSON>]
            }},
            "Summary": "<positive, ≤ 80‑word summary using only the Interpretation Dictionary>"
          }},
          "Heart_Line": {{ /* same structure */ }},
          "Life_Line":  {{ /* same structure */ }},
          "Fate_Line":  {{ /* same structure */ }},
          "variant": "<from input JSON>",
          "quality_issues": [ /* from input JSON */ ]
    }}
    
    Return exactly one JSON object matching the format above—no prose, no apologies.
    Please analyze the palm image according to the above instructions and return the interpretation in the JSON format shown. Use only positive, affirming language that highlights strengths, traits, and possibilities.
    """
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert palmistry analyst."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    #{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]}
            ],
            max_tokens=2000, 
            temperature=0.2 
        )
        if not response.choices or not response.choices[0].message or not response.choices[0].message.content:
            print("Warning: No content returned from GPT-4o.")
            return "Error: No content returned from GPT-4o."
        
        # The response content should be a JSON string, so we try to parse it
        response_content = response.choices[0].message.content.strip()
        
        # GPT might return the JSON block within triple backticks, remove them if present
        if response_content.startswith("```json"):
            response_content = response_content[7:]
        if response_content.endswith("```"):
            response_content = response_content[:-3]
            
        return json.loads(response_content) # Parse the JSON string into a Python dictionary

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response from GPT-4o: {e}")
        print(f"Raw response: {response_content}")
        return f"Error: Could not parse JSON response. Raw response: {response_content}"
    except Exception as e:
        print(f"An error occurred while calling OpenAI API: {e}")
        return f"Error: An API error occurred: {e}"

if __name__ == "__main__":
    s3_image_path = "s3://palm-reader/input-images/16 (1).jpg"  
    results = process_palm_image(s3_image_path)
    print("Processing complete. Output files:")
    print(json.dumps(results, indent=2))
