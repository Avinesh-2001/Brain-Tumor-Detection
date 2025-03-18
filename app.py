import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import tempfile
import os

# Set page configuration
st.set_page_config(page_title="Brain Tumor Detection", layout="wide")

# App title and description
st.title("Brain Tumor Detection using YOLO")
st.write("Upload an image to detect brain tumors using the YOLO model")

# Function to load the YOLO model
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

# Load the model
try:
    model = load_model("best.pt")
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Confidence threshold slider
conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)

if uploaded_file is not None:
    # Display the uploaded image
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Create a temporary file to save the uploaded image
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        tmp_file_path = tmp_file.name
        img_array = np.array(image)
        cv2.imwrite(tmp_file_path, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
    
    # Make prediction
    if st.button("Detect Brain Tumor"):
        with st.spinner("Analyzing image..."):
            # Run inference
            results = model.predict(tmp_file_path, conf=conf_threshold)
            
            # Plotting results
            with col2:
                st.subheader("Detection Results")
                # Get result image with bounding boxes
                result_image = Image.fromarray(results[0].plot())
                st.image(result_image, caption="Detection Result", use_container_width=True)
            
            # Display detection details
            st.subheader("Detection Details")
            
            if len(results[0].boxes) > 0:
                # Get the detection details
                boxes = results[0].boxes
                
                # Create a table for the detections
                data = []
                for i, box in enumerate(boxes):
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    
                    data.append({
                        "Detection #": i + 1,
                        "Class": class_name,
                        "Confidence": f"{confidence:.2f}",
                        "Bounding Box": box.xyxy[0].tolist()
                    })
                
                st.table(data)
                
                # Show summary
                st.success(f"Found {len(boxes)} tumor(s) in the image.")
            else:
                st.info("No tumors detected in the image.")
    
    # Clean up the temporary file
    os.unlink(tmp_file_path)