import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from qdrant_client import QdrantClient
import json
import io
import base64
from typing import List, Dict, Optional
import time
import plotly.graph_objects as go
import plotly.express as px

# Attempt to import timm for EfficientNet support
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Yoga Pose Similarity Search",
    page_icon="🧘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 1rem;
    }
    .stImage {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .result-card {
        background-color: #f7f7f7;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border: 1px solid #e0e0e0;
        transition: transform 0.2s;
    }
    .result-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    h1 {
        color: #1e3a8a;
        text-align: center;
        padding-bottom: 1rem;
        font-size: 2.5rem;
    }
    .upload-box {
        border: 2px dashed #4a5568;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #f8f9fa;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_config():
    """Load model configuration"""
    try:
        with open('model_config.json', 'r') as f:
            config = json.load(f)
            return config
    except FileNotFoundError:
        st.error("⚠️ model_config.json not found. Please ensure it's in the repository.")
        # Return default config
        return {
            "model_name": "resnet50",
            "embedding_dim": 2048,
            "collection_name": "yoga_poses",
            "total_images": 0,
            "unique_poses": [],
            "qdrant_url": ""
        }
    except json.JSONDecodeError:
        st.error("⚠️ Error reading model_config.json. Please check the file format.")
        return None

@st.cache_resource
def init_qdrant_client():
    """Initialize Qdrant client using Streamlit secrets"""
    try:
        # Get credentials from Streamlit secrets
        qdrant_url = st.secrets["QDRANT_URL"]
        qdrant_api_key = st.secrets["QDRANT_API_KEY"]
        
        client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            timeout=30  # Add timeout
        )
        
        # Test connection
        client.get_collections()
        
        return client
    except KeyError:
        st.error("❌ Qdrant credentials not found in Streamlit secrets.")
        st.info("Please add QDRANT_URL and QDRANT_API_KEY to your Streamlit app secrets.")
        return None
    except Exception as e:
        st.error(f"❌ Failed to connect to Qdrant: {str(e)}")
        return None

@st.cache_resource
def load_feature_extractor(model_name='resnet50'):
    """Load the feature extraction model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        if model_name == 'resnet50':
            # Using ResNet50
            model = models.resnet50(weights='IMAGENET1K_V1')
            model = nn.Sequential(*list(model.children())[:-1])
            embedding_dim = 2048
            
        elif model_name == 'efficientnet_b0' and TIMM_AVAILABLE:
            # Using EfficientNet
            model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
            embedding_dim = 1280
            
        elif model_name == 'vgg16':
            # Using VGG16
            model = models.vgg16(weights='IMAGENET1K_V1')
            model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
            embedding_dim = 4096
        else:
            # Default to ResNet50
            st.warning(f"Model {model_name} not available, using ResNet50 instead.")
            model = models.resnet50(weights='IMAGENET1K_V1')
            model = nn.Sequential(*list(model.children())[:-1])
            embedding_dim = 2048
        
        model = model.to(device)
        model.eval()
        
        # Image preprocessing pipeline
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        return model, transform, device, embedding_dim
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None

def extract_features(image, model, transform, device):
    """Extract features from an uploaded image"""
    try:
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Transform image
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        # Extract features
        with torch.no_grad():
            features = model(img_tensor)
            features = features.squeeze().cpu().numpy()
        
        # L2 normalize for better similarity search
        features = features / np.linalg.norm(features)
        
        return features
    
    except Exception as e:
        st.error(f"Error extracting features: {str(e)}")
        return None

def search_similar_poses(query_vector, client, collection_name, top_k=5):
    """Search for similar poses in Qdrant"""
    try:
        results = client.search(
            collection_name=collection_name,
            query_vector=query_vector.tolist(),
            limit=top_k
        )
        return results
    except Exception as e:
        st.error(f"Search failed: {str(e)}")
        st.info("Please check if the collection exists in Qdrant.")
        return []

def create_similarity_chart(results):
    """Create an interactive similarity chart"""
    if not results:
        return None
    
    # Extract data
    poses = []
    scores = []
    colors = []
    
    for i, r in enumerate(results):
        pose_name = r.payload.get('pose_name', 'Unknown')
        # Clean up pose name for display
        pose_name = pose_name.replace('_', ' ').title()
        poses.append(f"#{i+1} {pose_name}")
        scores.append(r.score)
        
        # Color gradient based on score
        if r.score > 0.9:
            colors.append('#10b981')  # Green
        elif r.score > 0.8:
            colors.append('#3b82f6')  # Blue
        elif r.score > 0.7:
            colors.append('#f59e0b')  # Orange
        else:
            colors.append('#ef4444')  # Red
    
    # Create horizontal bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=scores,
            y=poses,
            orientation='h',
            marker=dict(color=colors),
            text=[f"{s:.1%}" for s in scores],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Similarity: %{x:.2%}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title={
            'text': "Pose Similarity Scores",
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis=dict(
            title="Similarity Score",
            range=[0, 1.05],
            tickformat='.0%',
            gridcolor='rgba(0,0,0,0.1)'
        ),
        yaxis=dict(
            title="",
            autorange="reversed"
        ),
        height=400,
        margin=dict(l=20, r=20, t=40, b=40),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        )
    )
    
    return fig

def main():
    # Header
    st.markdown("# 🧘 Yoga Pose Similarity Search")
    st.markdown("##### Find similar yoga poses using AI-powered semantic search")
    st.markdown("---")
    
    # Load configuration
    config = load_config()
    if config is None:
        st.stop()
    
    # Initialize clients and models
    client = init_qdrant_client()
    if not client:
        st.error("Cannot proceed without Qdrant connection.")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📊 System Status")
        
        # System metrics in colored cards
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <h3>{config['total_images']}</h3>
                    <p>Total Images</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class="metric-card">
                    <h3>{len(config['unique_poses'])}</h3>
                    <p>Pose Classes</p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### ⚙️ Search Settings")
        top_k = st.slider(
            "Number of results to return",
            min_value=1,
            max_value=20,
            value=5,
            help="How many similar poses to find"
        )
        
        confidence_threshold = st.slider(
            "Minimum similarity score",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1,
            help="Filter results below this similarity score"
        )
        
        st.markdown("---")
        
        st.markdown("### 📝 Model Information")
        st.info(f"**Model:** {config['model_name']}")
        st.info(f"**Embedding Dim:** {config['embedding_dim']}")
        st.info(f"**Device:** {'GPU' if torch.cuda.is_available() else 'CPU'}")
        
        st.markdown("---")
        
        # Sample poses dropdown
        if config['unique_poses']:
            st.markdown("### 🔍 Browse Poses")
            selected_pose = st.selectbox(
                "Available poses in database:",
                [""] + sorted(config['unique_poses']),
                format_func=lambda x: x.replace('_', ' ').title() if x else "Select a pose..."
            )
            if selected_pose:
                st.write(f"Selected: **{selected_pose.replace('_', ' ').title()}**")
    
    # Main content area
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### 📤 Upload Your Image")
        
        # File uploader with custom styling
        uploaded_file = st.file_uploader(
            "Choose a yoga pose image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image of a yoga pose to find similar poses",
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Your uploaded image", use_column_width=True)
            
            # Get image info
            st.caption(f"Image size: {image.size[0]}x{image.size[1]} pixels")
            
            # Search button
            if st.button("🔍 Find Similar Poses", type="primary", use_container_width=True):
                
                # Load model
                with st.spinner("Loading AI model..."):
                    model, transform, device, embedding_dim = load_feature_extractor(config['model_name'])
                
                if model is None:
                    st.error("Failed to load model")
                    st.stop()
                
                # Extract features
                with st.spinner("Analyzing your image..."):
                    features = extract_features(image, model, transform, device)
                
                if features is None:
                    st.error("Failed to extract features")
                    st.stop()
                
                # Search
                with st.spinner("Searching database..."):
                    start_time = time.time()
                    results = search_similar_poses(
                        features,
                        client,
                        config['collection_name'],
                        top_k=top_k
                    )
                    search_time = time.time() - start_time
                
                # Filter by confidence threshold
                if confidence_threshold > 0:
                    results = [r for r in results if r.score >= confidence_threshold]
                
                # Store results
                st.session_state['search_results'] = results
                st.session_state['search_time'] = search_time
                
                # Success message
                st.success(f"✅ Found {len(results)} similar poses in {search_time:.2f} seconds!")
        else:
            # Empty state
            st.markdown("""
                <div class="upload-box">
                    <h4>👆 Upload an image to get started</h4>
                    <p>Supported formats: JPG, JPEG, PNG</p>
                </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🎯 Search Results")
        
        if 'search_results' in st.session_state and st.session_state['search_results']:
            results = st.session_state['search_results']
            search_time = st.session_state.get('search_time', 0)
            
            # Performance metric
            st.metric("Search Time", f"{search_time:.3f}s")
            
            # Similarity chart
            fig = create_similarity_chart(results)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed results
            st.markdown("#### 📋 Detailed Results")
            
            for idx, result in enumerate(results):
                pose_name = result.payload.get('pose_name', 'Unknown')
                clean_name = pose_name.replace('_', ' ').title()
                score = result.score
                
                # Create expandable result cards
                with st.expander(f"#{idx+1} - {clean_name} ({score:.1%})", expanded=(idx==0)):
                    cols = st.columns([2, 1])
                    
                    with cols[0]:
                        st.markdown(f"**Pose Name:** {clean_name}")
                        st.markdown(f"**Similarity Score:** {score:.2%}")
                        st.markdown(f"**Rank:** #{idx+1} out of {config['total_images']} images")
                    
                    with cols[1]:
                        # Score indicator
                        if score > 0.9:
                            st.success("Very High Match")
                        elif score > 0.8:
                            st.info("High Match")
                        elif score > 0.7:
                            st.warning("Medium Match")
                        else:
                            st.error("Low Match")
                    
                    # Additional metadata
                    if result.payload.get('image_filename'):
                        st.caption(f"Source: {result.payload['image_filename']}")
                    
                    # Show all metadata in debug mode
                    with st.expander("View metadata"):
                        st.json(result.payload)
        
        else:
            # Empty state for results
            st.info("👈 Upload an image and click 'Find Similar Poses' to see results")
            
            # Show sample search
            st.markdown("---")
            st.markdown("#### 💡 How it works")
            st.markdown("""
            1. **Upload** a yoga pose image
            2. **AI extracts** visual features using deep learning
            3. **Vector search** finds similar poses in milliseconds
            4. **Results ranked** by visual similarity
            """)
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Built with:**")
        st.caption("Streamlit • PyTorch • Qdrant")
    with col2:
        st.markdown("**Dataset:**")
        st.caption("107 Yoga Poses Dataset")
    with col3:
        st.markdown("**Course:**")
        st.caption("Advanced Computer Vision")

if __name__ == "__main__":
    main()
