# fastapi_backend/main.py

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import torch
from torchvision import transforms
from PIL import Image, ImageEnhance, ImageFilter
import io
import base64
import os
import numpy as np
import cv2
import mediapipe as mp
from model_class.model_0 import TinyVGG, INPUT_SHAPE, HIDDEN_UNITS, OUTPUT_SHAPE


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your trained model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "models", "model_0.pth")

model = TinyVGG(input_shape=INPUT_SHAPE, hidden_units=HIDDEN_UNITS, output_shape=OUTPUT_SHAPE)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
print(f"Model loaded successfully")
model.eval()

# Enhanced preprocessing pipeline for better hand detection
def enhance_image_for_hand_detection(img):
    """
    Apply multiple enhancement techniques to improve hand detection
    """
    # Convert to RGB if not already
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.5)
    
    # Enhance sharpness
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.3)
    
    # Apply edge-preserving smoothing
    img = img.filter(ImageFilter.SMOOTH_MORE)
    
    # Apply unsharp mask for better edge definition
    img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    
    return img

def remove_background_opencv(img):
    """
    Advanced background removal using OpenCV
    """
    try:
        # Convert PIL to OpenCV format
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Apply bilateral filter to reduce noise while preserving edges
        img_cv = cv2.bilateralFilter(img_cv, 9, 75, 75)
        
        # Convert to HSV for better skin detection
        hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
        
        # Define skin color range in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create mask for skin color
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Apply morphological operations to clean the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Apply Gaussian blur to soften edges
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        # Convert mask to 3 channels
        mask_3d = cv2.merge([mask, mask, mask])
        
        # Apply mask
        result = cv2.bitwise_and(img_cv, mask_3d)
        
        # Convert back to PIL format
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result_rgb)
        
    except Exception as e:
        print(f"OpenCV processing failed: {e}")
        # Fallback to simple method
        return remove_background_simple(img)

def remove_background_simple(img):
    """
    Simple background removal using thresholding and morphological operations
    """
    # Convert PIL to numpy array
    img_array = np.array(img)
    
    # Create a mask for skin tones (rough approximation)
    # This is a simple method - for better results, use proper hand detection models
    r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
    
    # Skin tone detection mask
    skin_mask = (
        (r > 95) & (g > 40) & (b > 20) &
        (np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b]) > 15) &
        (np.abs(r.astype(int) - g.astype(int)) > 15) &
        (r > g) & (r > b)
    )
    
    # Apply morphological operations to clean up the mask
    # Convert boolean mask to uint8
    skin_mask = skin_mask.astype(np.uint8) * 255
    
    # Create a 3-channel version of the mask
    skin_mask_3d = np.stack([skin_mask, skin_mask, skin_mask], axis=2)
    
    # Apply mask to original image
    result = np.where(skin_mask_3d > 0, img_array, 0)
    
    # Convert back to PIL Image
    return Image.fromarray(result.astype(np.uint8))

# EXACT same transform as training - NO normalization!
data_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.ToTensor(),
])

labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["Space", "Delete", "Nothing"]

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def detect_hand_and_create_white_background(image_array):
    """
    Detect hand using MediaPipe and create a clean white background.
    This should significantly improve model accuracy by providing consistent backgrounds.
    """
    try:
        # Convert numpy array to OpenCV format
        if isinstance(image_array, np.ndarray):
            if image_array.dtype != np.uint8:
                image_array = (image_array * 255).astype(np.uint8) if image_array.max() <= 1.0 else image_array.astype(np.uint8)
            cv_image = image_array
        else:
            # Convert PIL to OpenCV
            cv_image = cv2.cvtColor(np.array(image_array), cv2.COLOR_RGB2BGR)
        
        # Initialize MediaPipe hands
        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        ) as hands:
            
            # Convert BGR to RGB for MediaPipe
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            # Process the image
            results = hands.process(rgb_image)
            
            # Create white background
            height, width = rgb_image.shape[:2]
            white_bg = np.ones((height, width, 3), dtype=np.uint8) * 255
            
            if results.multi_hand_landmarks:
                # Hand detected - create mask
                for hand_landmarks in results.multi_hand_landmarks:
                    # Get hand landmarks
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        x = int(lm.x * width)
                        y = int(lm.y * height)
                        landmarks.append([x, y])
                    
                    # Create convex hull around hand
                    landmarks = np.array(landmarks)
                    hull = cv2.convexHull(landmarks)
                    
                    # Create mask with some padding around the hand
                    mask = np.zeros((height, width), dtype=np.uint8)
                    cv2.fillPoly(mask, [hull], 255)
                    
                    # Dilate mask to include more of the hand/fingers
                    kernel = np.ones((20, 20), np.uint8)
                    mask = cv2.dilate(mask, kernel, iterations=1)
                    
                    # Apply mask to original image with white background
                    for c in range(3):
                        white_bg[:, :, c] = np.where(mask == 255, rgb_image[:, :, c], white_bg[:, :, c])
                
                print("Hand detected and background replaced with white")
                return Image.fromarray(white_bg)
            else:
                print("No hand detected, using original image with white background")
                # If no hand detected, still try to create a rough mask based on color
                return create_rough_hand_mask_with_white_bg(rgb_image)
                
    except Exception as e:
        print(f"Hand detection failed: {e}")
        # Fallback to original image
        if isinstance(image_array, np.ndarray):
            return Image.fromarray(image_array)
        else:
            return image_array

def create_rough_hand_mask_with_white_bg(rgb_image):
    """
    Fallback method to create rough hand mask when MediaPipe fails to detect hands
    """
    try:
        # Convert to HSV for better skin detection
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        
        # Define skin color range in HSV
        lower_skin1 = np.array([0, 20, 70])
        upper_skin1 = np.array([20, 255, 255])
        
        lower_skin2 = np.array([160, 20, 70])
        upper_skin2 = np.array([180, 255, 255])
        
        # Create mask for skin tones
        mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
        mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find largest contour (likely the hand)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(mask)
            cv2.fillPoly(mask, [largest_contour], 255)
        
        # Create white background
        height, width = rgb_image.shape[:2]
        white_bg = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Apply mask
        for c in range(3):
            white_bg[:, :, c] = np.where(mask == 255, rgb_image[:, :, c], white_bg[:, :, c])
        
        print("Created rough hand mask with white background")
        return Image.fromarray(white_bg)
        
    except Exception as e:
        print(f"Rough hand mask creation failed: {e}")
        return Image.fromarray(rgb_image)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection accepted")
    while True:
        try:
            data = await websocket.receive_text()
            print(f"Received data length: {len(data)}")
            
            # Check if data has the base64 prefix
            if "," in data:
                img_data = base64.b64decode(data.split(",")[1])
            else:
                # If no prefix, assume it's pure base64
                img_data = base64.b64decode(data)
            
            print(f"Decoded image data length: {len(img_data)}")            # Load and preprocess image
            img = Image.open(io.BytesIO(img_data)).convert("RGB")
            print(f"Original image size: {img.size}")
            
            # Apply hand detection and white background
            processed_img = detect_hand_and_create_white_background(img)
            print(f"Processed image size: {processed_img.size}")
            
            # Use EXACT same preprocessing as training (resize to 64x64 + normalize)
            try:
                img_tensor = data_transform(processed_img).unsqueeze(0)
                print(f"Preprocessing - tensor shape: {img_tensor.shape}")
                print(f"Tensor min: {img_tensor.min():.4f}, max: {img_tensor.max():.4f}")
                print(f"Tensor mean: {img_tensor.mean():.4f}, std: {img_tensor.std():.4f}")
                
                # Model inference
                with torch.no_grad():
                    outputs = model(img_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)
                    
                    confidence_score = confidence.item()
                    label = labels[predicted.item()]
                    
                    print(f"Prediction: {label} (confidence: {confidence_score:.3f})")
                    print(f"Top 3 predictions:")
                    top3_probs, top3_indices = torch.topk(probabilities, 3)
                    for i in range(3):
                        idx = top3_indices[0][i].item()
                        prob = top3_probs[0][i].item()
                        print(f"  {i+1}. {labels[idx]}: {prob:.3f}")
                
            except Exception as preprocessing_error:
                print(f"Preprocessing failed: {preprocessing_error}")
                import traceback
                traceback.print_exc()
                
                # Fallback to absolute minimal preprocessing
                img_resized = processed_img.resize((64, 64))
                img_array = np.array(img_resized).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
                print(f"Fallback preprocessing - tensor shape: {img_tensor.shape}")
                
                with torch.no_grad():
                    outputs = model(img_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)
                    
                    confidence_score = confidence.item()
                    label = labels[predicted.item()]
                    
                    print(f"Fallback prediction: {label} (confidence: {confidence_score:.3f})")# Model inference - use the results from above
            # Send prediction back with more context
            if confidence_score > 0.3:  # Lower threshold for testing
                response = f"{label} ({confidence_score:.2f})"
            else:
                response = f"Uncertain ({confidence_score:.2f})"

            # Send prediction back
            await websocket.send_text(response)
            
        except Exception as e:
            print(f"Error processing image: {e}")
            import traceback
            traceback.print_exc()
            try:
                await websocket.send_text("Error")
            except:
                break
                break
        except KeyboardInterrupt:
            break
