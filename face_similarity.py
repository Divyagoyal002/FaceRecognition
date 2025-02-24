import streamlit as st
import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms
from PIL import Image
from scipy.spatial.distance import cosine, mahalanobis

# Load MTCNN for face detection
detector = MTCNN(keep_all=True)

# Load VGGFace model for feature extraction
model = InceptionResnetV1(pretrained='vggface2').eval()

def detect_faces(image):
    """Detect faces in an image using MTCNN and return cropped faces."""
    boxes, _ = detector.detect(image)
    faces = []
    if boxes is not None:
        for box in boxes:
            x, y, w, h = map(int, box)
            face = image[y:h, x:w]
            faces.append(face)
    return faces

def preprocess_face(face):
    """Resize and normalize face images for model input."""
    face = cv2.resize(face, (160, 160))
    face = Image.fromarray(face)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(face).unsqueeze(0)

def extract_embedding(face):
    """Extract face embeddings using the VGGFace model."""
    with torch.no_grad():
        embedding = model(face)
    return embedding.numpy().flatten()

def calculate_similarity(embedding1, embedding2, inv_cov_matrix, cosine_weight=0.7, mahal_weight=0.3, threshold=0.5):
    """Hybrid similarity using Cosine and Mahalanobis distance."""
    cosine_sim = 1 - cosine(embedding1, embedding2)
    mahalanobis_dist = mahalanobis(embedding1, embedding2, inv_cov_matrix)
    mahal_scaled = np.exp(-mahalanobis_dist)  # Convert distance to similarity
    
    hybrid_score = (cosine_weight * cosine_sim) + (mahal_weight * mahal_scaled)
    return hybrid_score, hybrid_score > threshold

st.title("Face Similarity Detector")

uploaded_file1 = st.file_uploader("Upload First Image", type=["jpg", "png", "jpeg"])
uploaded_file2 = st.file_uploader("Upload Second Image", type=["jpg", "png", "jpeg"])

if uploaded_file1 and uploaded_file2:
    image1 = np.array(Image.open(uploaded_file1).convert("RGB"))
    image2 = np.array(Image.open(uploaded_file2).convert("RGB"))
    
    faces1 = detect_faces(image1)
    faces2 = detect_faces(image2)

    if not faces1 or not faces2:
        st.write("Could not detect faces in one or both images.")
    else:
        found_similar = 0
        all_faces = faces1 + faces2

        embeddings = [extract_embedding(preprocess_face(face)) for face in all_faces]
        inv_cov_matrix = np.linalg.pinv(np.cov(np.vstack(embeddings).T))  # More stable inverse

        for face1 in faces1:
            face1_tensor = preprocess_face(face1)
            embedding1 = extract_embedding(face1_tensor)

            for face2 in faces2:
                face2_tensor = preprocess_face(face2)
                embedding2 = extract_embedding(face2_tensor)

                similarity, is_same = calculate_similarity(embedding1, embedding2, inv_cov_matrix)
                if is_same:
                    found_similar += 1

        st.image([image1, image2], caption=["Image 1", "Image 2"], width=300)
        st.write(f"Similar Faces Found: {found_similar}")

        