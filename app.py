import streamlit as st
from PIL import Image, ImageEnhance
from transformers import pipeline
import json
import time

st.set_page_config(
    page_title="FischID • Next Generation",
    page_icon="🐟",
    layout="centered"
)

# Modernes Design
st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #0a0f1c 0%, #1e2937 100%); color: #e0f2fe;}
    h1 {font-size: 4rem; background: linear-gradient(90deg, #22d3ee, #a855f7, #ec4899);
         -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center;}
    .subtitle {text-align: center; color: #94a3b8; font-size: 1.5rem;}
    .result {background: linear-gradient(90deg, #10b981, #34d399); color: white; 
             border-radius: 20px; padding: 2rem;}
    .timer {color: #67e8f9; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

st.title("FischID")
st.markdown('<p class="subtitle">Next Generation KI-Fisch-Erkennung</p>', unsafe_allow_html=True)

# ====================== MODELL ======================
@st.cache_resource(show_spinner="Lade KI-Modell...")
def load_model():
    try:
        return pipeline(
            "image-classification", 
            model="google/vit-base-patch16-224", 
            top_k=6,
            device=-1  # CPU erzwingen
        )
    except:
        st.error("Modell konnte nicht geladen werden. Bitte App neu starten.")
        st.stop()

classifier = load_model()

# fish_data.json
with open("fish_data.json", "r", encoding="utf-8") as f:
    fish_data = json.load(f)

# ====================== FOTO ======================
st.subheader("📸 Foto aufnehmen oder hochladen")

col1, col2 = st.columns(2)
with col1:
    camera = st.camera_input("Kamera")
with col2:
    uploaded = st.file_uploader("Bild hochladen", type=["jpg", "jpeg", "png"])

if camera or uploaded:
    image = Image.open(camera if camera else uploaded).convert("RGB")
    
    # Bild verbessern
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.3)
    
    st.image(image, caption="Dein Foto", use_column_width=True)

    # Schöner Fortschritt
    progress_bar = st.progress(0)
    status_text = st.empty()
    start_time = time.time()

    for i in range(100):
        time.sleep(0.015)
        progress_bar.progress(i + 1)
        elapsed = time.time() - start_time
        status_text.markdown(f"<p class='timer'>KI analysiert... {elapsed:.2f} s</p>", unsafe_allow_html=True)

    # Vorhersage
    with st.spinner("KI trifft Entscheidung..."):
        results = classifier(image)

    top = results[0]
    label = top['label'].replace("_", " ").title()
    confidence = top['score'] * 100

    # Mapping
    mapping = {
        "Pike": "Hecht", "Zander": "Zander", "Perch": "Flussbarsch", "Carp": "Karpfen",
        "Trout": "Meerforelle", "Bream": "Brassen", "Catfish": "Wels", "Eel": "Aal",
        "Flatfish": "Scholle", "Roach": "Rotauge"
    }
    fish_name = mapping.get(label, label)

    total_time = time.time() - start_time

    if confidence >= 75:
        st.markdown('<div class="result">', unsafe_allow_html=True)
        st.success(f"**Erkannte Art:** {fish_name}")
        st.success(f"**Sicherheit:** {confidence:.1f}%")
        st.caption(f"⏱ Analysezeit: {total_time:.2f} Sekunden")
        st.markdown('</div>', unsafe_allow_html=True)

        bundesland = st.selectbox("🌍 Bundesland", list(fish_data["bundeslaender"].keys()))

        info = fish_data["bundeslaender"][bundesland].get(fish_name, {})
        if info:
            c1, c2 = st.columns(2)
            with c1: st.metric("Mindestmaß", f"{info.get('mindestmass', 0)} cm")
            with c2: st.metric("Schonzeit", info.get('schonzeit', "Keine"))
    else:
        st.warning(f"Nur {confidence:.1f}% Sicherheit. Bitte ein klareres Foto versuchen.")

st.markdown("---")
st.caption("FischID • Google ViT • Modernes Design")
