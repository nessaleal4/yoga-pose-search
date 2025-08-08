import streamlit as st
from qdrant_client import QdrantClient
import json
from PIL import Image
import numpy as np
import base64
import time
import plotly.graph_objects as go

# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Yoga Pose Similarity Search",
    page_icon="🧘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main { padding: 1rem; }
    .upload-box {
        border: 2px dashed #4a5568;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #f8f9fa;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; padding: 1rem; border-radius: 10px; text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Config Loader
# ─────────────────────────────────────────────
@st.cache_resource
def load_config():
    try:
        with open('model_config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("⚠️ model_config.json not found.")
        return {
            "collection_name": "yoga_poses",
            "total_images": 0,
            "unique_poses": [],
            "qdrant_url": ""
        }

# ─────────────────────────────────────────────
# Qdrant Client
# ─────────────────────────────────────────────
@st.cache_resource
def init_qdrant_client():
    try:
        client = QdrantClient(
            url=st.secrets["QDRANT_URL"],
            api_key=st.secrets["QDRANT_API_KEY"],
            timeout=30
        )
        client.get_collections()  # Test connection
        return client
    except Exception as e:
        st.error(f"❌ Failed to connect to Qdrant: {str(e)}")
        return None

# ─────────────────────────────────────────────
# Search Function
# ─────────────────────────────────────────────
def search_similar_poses(query_vector, client, collection_name, top_k=5):
    try:
        results = client.search(
            collection_name=collection_name,
            query_vector=query_vector.tolist(),
            limit=top_k
        )
        return results
    except Exception as e:
        st.error(f"Search failed: {str(e)}")
        return []

# ─────────────────────────────────────────────
# Chart
# ─────────────────────────────────────────────
def create_similarity_chart(results):
    if not results:
        return None
    poses = []
    scores = []
    colors = []
    for i, r in enumerate(results):
        pose_name = r.payload.get('pose_name', 'Unknown').replace('_', ' ').title()
        poses.append(f"#{i+1} {pose_name}")
        scores.append(r.score)
        colors.append('#10b981' if r.score > 0.9 else '#3b82f6' if r.score > 0.8 else '#f59e0b' if r.score > 0.7 else '#ef4444')

    fig = go.Figure(data=[
        go.Bar(
            x=scores, y=poses, orientation='h', marker=dict(color=colors),
            text=[f"{s:.1%}" for s in scores], textposition='outside'
        )
    ])
    fig.update_layout(
        title="Pose Similarity Scores", xaxis=dict(range=[0, 1.05], tickformat='.0%'),
        yaxis=dict(autorange="reversed"), height=400
    )
    return fig

# ─────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────
def main():
    st.title("🧘 Yoga Pose Similarity Search")
    config = load_config()
    client = init_qdrant_client()
    if not client:
        st.stop()

    # Sidebar
    with st.sidebar:
        st.markdown("## 📊 System Status")
        col1, col2 = st.columns(2)
        col1.markdown(f"<div class='metric-card'><h3>{config['total_images']}</h3><p>Total Images</p></div>", unsafe_allow_html=True)
        col2.markdown(f"<div class='metric-card'><h3>{len(config['unique_poses'])}</h3><p>Pose Classes</p></div>", unsafe_allow_html=True)
        st.markdown("---")
        top_k = st.slider("Number of results", 1, 20, 5)
        confidence_threshold = st.slider("Min similarity", 0.0, 1.0, 0.0, 0.1)
        st.markdown("---")

    # Upload
    uploaded_file = st.file_uploader("Upload a yoga pose image", type=['jpg', 'jpeg', 'png'])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # 🔹 Here is where you'd get the embedding from an API instead of computing locally
        st.info("Currently, embedding extraction is skipped. Replace this step with an API call.")
        # Example placeholder embedding (must match dimension of your Qdrant vectors)
        placeholder_embedding = np.random.rand(config['embedding_dim'])

        if st.button("🔍 Find Similar Poses"):
            with st.spinner("Searching Qdrant..."):
                start = time.time()
                results = search_similar_poses(placeholder_embedding, client, config['collection_name'], top_k=top_k)
                results = [r for r in results if r.score >= confidence_threshold]
                st.session_state['search_results'] = results
                st.session_state['search_time'] = time.time() - start
            st.success(f"Found {len(results)} results in {st.session_state['search_time']:.2f}s")

    # Results
    if 'search_results' in st.session_state:
        results = st.session_state['search_results']
        if results:
            fig = create_similarity_chart(results)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            for idx, r in enumerate(results):
                st.write(f"**#{idx+1} - {r.payload.get('pose_name', 'Unknown')}** ({r.score:.1%})")

if __name__ == "__main__":
    main()
