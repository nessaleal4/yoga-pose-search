# Yoga Pose Similarity Search 

An AI-powered web application for finding similar yoga poses using semantic image search.

## Features
- Upload yoga pose images
- Find similar poses using deep learning embeddings
- Interactive visualization of similarity scores
- Real-time search powered by Qdrant vector database

## Technology Stack
- **Frontend**: Streamlit
- **Deep Learning**: PyTorch, ResNet50
- **Vector Database**: Qdrant Cloud
- **Deployment**: Streamlit Cloud

## Live Demo
[View Live Application](https://YOUR-APP-NAME.streamlit.app)

## Assignment Details
- Course: Advanced Computer Vision
- Assignment: 3 - Semantic Search
- Dataset: [Yoga Poses Dataset (107 poses)](https://www.kaggle.com/datasets/arrowe/yoga-poses-dataset-107)

## Architecture
1. Pre-trained CNN (ResNet50) extracts image embeddings
2. Embeddings stored in Qdrant vector database
3. Cosine similarity used for finding similar poses
4. Streamlit provides interactive web interface

## Local Development
1. Clone repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up Qdrant credentials in `.streamlit/secrets.toml`
4. Run: `streamlit run app.py`

## Authors
Vanessa leal
