import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

class CustomPyTorchGazeTracker:
    def __init__(self, model_filename="mobilenetv2_clean_best.pth"):
        # We enforce CPU for Flask so multiple threads don't crash CUDA
        self.device = torch.device("cpu")
        self.classes = ['looking_down', 'looking_left', 'looking_right', 
                        'looking_straight', 'looking_up', 'multiple_faces']
        
        # Dynamically resolve paths for cloud deployment
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, "models", model_filename)
        
        print(f"[PyTorch Gaze Tracker] Loading model from {model_path} onto {self.device}...")
        
        # Load Architecture
        self.model = models.mobilenet_v2(weights=None)
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_ftrs, len(self.classes))
        
        # Load Weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Standardize Image Transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Initialize Haar Cascades for Face Cropping (fallback)
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        print("[PyTorch Gaze Tracker] Ready for inference.")

    def predict_gaze(self, frame):
        """
        Expects a BGR frame (from OpenCV).
        Finds the face, crops it, runs it through MobileNetV2.
        Returns the gaze direction as a string mapped to the old system's grammar.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        if len(faces) == 0:
            return "unknown"
        elif len(faces) > 1:
            return "multiple_faces"  # Note: your YOLO handles multi-person already, but this is a nice fallback
            
        x, y, w, h = faces[0]
        crop = frame[y:y+h, x:x+w]
        
        # Convert BGR (OpenCV) to RGB (PIL)
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(crop_rgb)
        
        # Apply transforms and add batch dimension
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
            _, predicted_idx = torch.max(outputs, 1)
            predicted_class = self.classes[predicted_idx.item()]
            
        # Map class name to existing MediaPipe grammar 
        # (your test script has: 'looking_down', 'looking_left', 'looking_right', 'looking_straight', 'looking_up')
        if predicted_class == "looking_straight":
            return "forward"
        elif predicted_class == "looking_down":
            return "down"
        elif predicted_class == "looking_up":
            return "up"
        elif predicted_class == "looking_left":
            return "left"
        elif predicted_class == "looking_right":
            return "right"
        else:
            return "unknown"
