# Yoga Pose Similarity Search
An AI-powered web application for finding similar yoga poses using semantic image search with precomputed embeddings.

## Features
- Upload yoga pose images  
- Retrieve visually similar poses from a pre-indexed database in **Qdrant Cloud**  
- Interactive visualization of similarity scores  
- Fast search without heavy model inference on Streamlit  

## Technology Stack
- **Frontend:** Streamlit  
- **Vector Database:** Qdrant Cloud  
- **Deployment:** Streamlit Cloud  

## Live Demo
[[View Live Application](#) ](https://yoga-pose-search.streamlit.app/)

## Assignment Details
- **Course:** Advanced Computer Vision with Deep Learning – University of Chicago  
- **Assignment:** 3 – Semantic Search  
- **Dataset:** Yoga Poses Dataset (107 poses): https://www.kaggle.com/datasets/arrowe/yoga-poses-dataset-107

## Architecture
1. **Offline:**  
   - Pre-trained CNN (e.g., ResNet50) used locally to extract embeddings for all yoga images  
   - Embeddings uploaded and stored in Qdrant Cloud  
2. **Online (Streamlit app):**  
   - User uploads an image  
   - App queries Qdrant for similar embeddings (no on-device model computation)  
   - Results displayed with similarity scores and interactive chart  

## Local Development
1. Clone repository  
```bash
git clone https://github.com/nessaleal4/yoga-pose-search.git
cd yoga-pose-search
```  

2. Install dependencies  
```bash
pip install -r requirements.txt
```  

3. Create `.streamlit/secrets.toml` with your Qdrant credentials:  
```toml
QDRANT_URL = "your-qdrant-url"
QDRANT_API_KEY = "your-api-key"
```  

4. Run the app:  
```bash
streamlit run app.py
```  

## Author
Vanessa Leal Atwood
