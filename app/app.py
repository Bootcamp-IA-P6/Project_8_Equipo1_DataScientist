import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from supabase import create_client, Client
from PIL import Image
from datetime import datetime

# ─────────────────────────────────────────────
# 1. CONFIGURACIÓN Y CARGA DE SECRETOS (.env)
# ─────────────────────────────────────────────
load_dotenv() # Carga las variables de tu archivo .env

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Inicializar Supabase (con manejo de errores)
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception:
    supabase = None

# ─────────────────────────────────────────────
# 2. CONFIGURACIÓN DE PÁGINA Y ESTILOS CSS
# ─────────────────────────────────────────────
st.set_page_config(page_title="Stroke Risk AI", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    /* Importar fuente Inter */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #F8FAFC !important;
    }

    /* Tarjetas Médicas */
    .med-card {
        background: white;
        border-radius: 12px;
        padding: 24px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
    }

    /* KPIs con bordes de colores (Mismo estilo que screenshot 1) */
    .kpi-box { border-top: 4px solid #006699; background: white; padding: 20px; border-radius: 8px; }
    .kpi-red { border-top-color: #E11D48; }
    .kpi-orange { border-top-color: #F59E0B; }

    /* Estilo de la Navegación (Tabs) */
    .stTabs [data-baseweb="tab-list"] { gap: 24px; background-color: transparent; }
    .stTabs [data-baseweb="tab"] {
        height: 40px; white-space: pre; background-color: transparent;
        border: none; color: #64748B; font-weight: 500;
    }
    .stTabs [aria-selected="true"] { color: #006699 !important; border-bottom: 2px solid #006699 !important; }

    /* Sidebar Derecho (Patient Session) */
    .sidebar-right {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px;
    }

    /* Botones */
    .stButton > button { border-radius: 8px; font-weight: 600; height: 45px; }
    
    /* Riesgo Alto (Texto Rojo) */
    .risk-critical { color: #E11D48; font-size: 48px; font-weight: 800; margin: 10px 0; }
    
    /* Alertas */
    .alert-box {
        background: #FFF1F2; border: 1px solid #FECDD3;
        padding: 15px; border-radius: 10px; color: #9F1239; font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 3. LÓGICA DE NAVEGACIÓN
# ─────────────────────────────────────────────
if 'page' not in st.session_state:
    st.session_state.page = "Directory"

def render_navbar():
    col1, col2, col3 = st.columns([3, 5, 3])
    with col1:
        st.markdown("<h3 style='color:#0F172A; margin:0;'>🧠 Stroke Risk <span style='font-weight:300;'>Assessment</span></h3>", unsafe_allow_html=True)
    with col2:
        # Botones de navegación tipo Tab
        c1, c2, c3, c4 = st.columns(4)
        if c1.button("Directory", type="secondary" if st.session_state.page != "Directory" else "primary", use_container_width=True):
            st.session_state.page = "Directory"
            st.rerun()
        if c2.button("Patient Data", type="secondary" if st.session_state.page != "Data" else "primary", use_container_width=True):
            st.session_state.page = "Data"
            st.rerun()
        if c3.button("CT Scan", type="secondary" if st.session_state.page != "CT" else "primary", use_container_width=True):
            st.session_state.page = "CT"
            st.rerun()
        if c4.button("Results", type="secondary" if st.session_state.page != "Results" else "primary", use_container_width=True):
            st.session_state.page = "Results"
            st.rerun()
    with col3:
        st.button("+ New Patient", type="primary", use_container_width=True)

# ─────────────────────────────────────────────
# 4. COMPONENTE SIDEBAR DERECHO
# ─────────────────────────────────────────────
def right_sidebar():
    st.markdown("""
    <div class="sidebar-right">
        <p style="color:#64748B; font-size:11px; font-weight:700; margin-bottom:15px;">PATIENT SESSION</p>
        <div style="display:flex; align-items:center; gap:12px; margin-bottom:20px;">
            <div style="width:45px; height:45px; background:#F1F5F9; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:20px;">👤</div>
            <div>
                <div style="font-weight:700; color:#1E293B; font-size:14px;">Eleanor Vance</div>
                <div style="color:#64748B; font-size:12px;">ID: 294-881-00</div>
            </div>
        </div>
        <div style="font-size:13px; color:#475569; border-top: 1px solid #F1F5F9; padding-top:10px; line-height:2.5;">
            <div style="color:#006699; font-weight:700;">📊 Risk Summary</div>
            <div>🕒 Clinical History</div>
            <div>🫀 Vitals</div>
        </div>
        <div style="margin-top:100px; background:#F8FAFC; padding:10px; border-radius:8px;">
            <p style="font-size:10px; font-weight:700; margin:0;">CURRENT VITALS</p>
            <div style="display:flex; justify-content:space-between; font-size:11px; margin-top:5px;">
                <span>BP: 142/90</span>
                <span>HR: 78 bpm</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    st.button("Export Report", use_container_width=True)

# ─────────────────────────────────────────────
# PÁGINA 1: DIRECTORY
# ─────────────────────────────────────────────
def show_directory():
    st.markdown("## Patient Directory")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown('<div class="kpi-box"><small>TOTAL ANALYZED</small><h2>1,284</h2><p style="color:green; font-size:12px;">↗ +12% vs last month</p></div>', unsafe_allow_html=True)
    with c2: st.markdown('<div class="kpi-box kpi-red"><small>HIGH RISK CASES</small><h2 style="color:#E11D48;">14</h2><p style="color:gray; font-size:12px;">Immediate review</p></div>', unsafe_allow_html=True)
    with c3: st.markdown('<div class="kpi-box kpi-orange"><small>MEDIUM RISK</small><h2 style="color:#F59E0B;">42</h2><p style="color:gray; font-size:12px;">Monitoring</p></div>', unsafe_allow_html=True)
    with c4: st.markdown('<div class="kpi-box"><small>PENDING SCANS</small><h2>8</h2><p style="color:gray; font-size:12px;">In queue</p></div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="med-card">', unsafe_allow_html=True)
    df = pd.DataFrame({
        "PATIENT ID": ["PT-2023-001", "PT-2023-002", "PT-2023-003", "PT-2023-004"],
        "PATIENT NAME": ["Sarah Jenkins", "Michael Ross", "Eleanor Wright", "David Kim"],
        "AGE": [68, 54, 72, 45],
        "LAST ANALYSIS": ["Oct 24, 2023", "Oct 23, 2023", "Oct 22, 2023", "Oct 21, 2023"],
        "RISK LEVEL": ["High", "Medium", "Low", "Low"]
    })
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PÁGINA 2: PATIENT DATA (ML Model)
# ─────────────────────────────────────────────
def show_data_entry():
    col_main, col_side = st.columns([8, 2.5])
    with col_main:
        st.markdown("## Patient Data Entry")
        st.markdown("<p style='color:#64748B;'>Complete the comprehensive risk profile for diagnostic modeling.</p>", unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="med-card"><b>👤 Patient Information</b><br><br>', unsafe_allow_html=True)
            st.selectbox("Gender", ["Female", "Male"])
            st.number_input("Age (years)", value=67)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="med-card"><b>📈 Clinical Metrics</b><br><br>', unsafe_allow_html=True)
            st.number_input("Avg. Glucose Level (mg/dL)", value=228.69)
            st.number_input("BMI (kg/m²)", value=36.6)
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="med-card"><b>🏥 Medical History</b><br><br>', unsafe_allow_html=True)
            st.toggle("Hypertension", value=True)
            st.toggle("Heart Disease", value=False)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="med-card"><b>🌍 Demographics & Lifestyle</b><br><br>', unsafe_allow_html=True)
            st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes"])
            st.selectbox("Work Type", ["Private", "Self-employed", "Government"])
            st.markdown('</div>', unsafe_allow_html=True)

        if st.button("Calculate Stroke Risk", type="primary", use_container_width=True):
            st.toast("Running ML Supervised Model...")

    with col_side:
        st.markdown('<div class="med-card" style="text-align:center;">', unsafe_allow_html=True)
        st.markdown("<small>RISK ASSESSMENT</small> <span style='float:right; color:#E11D48; background:#FFF1F2; padding:2px 6px; border-radius:4px; font-size:10px;'>HIGH ALERT</span>", unsafe_allow_html=True)
        st.markdown("<h1 class='risk-critical'>82.4%</h1>", unsafe_allow_html=True)
        st.markdown("<b>Stroke Probability</b>", unsafe_allow_html=True)
        st.progress(0.64)
        st.markdown("<p style='font-size:11px; color:gray; text-align:left;'>Model Confidence: 64.2%</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="alert-box">⚠️ <b>Model uncertainty detected.</b><br>CT scan recommended for validation.</div>', unsafe_allow_html=True)
        st.write("")
        right_sidebar()

# ─────────────────────────────────────────────
# PÁGINA 3: CT SCAN (CNN Model)
# ─────────────────────────────────────────────
def show_ct_scan():
    col_main, col_side = st.columns([8, 2.5])
    with col_main:
        st.markdown("## CT Scan Analysis")
        
        up, res = st.columns([1, 1])
        with up:
            st.markdown('<div class="med-card" style="border: 2px dashed #CBD5E1; text-align:center; padding:40px;">', unsafe_allow_html=True)
            st.write("📄")
            st.markdown("<b>Upload CT Scan</b><br><small>DICOM, PNG, or JPG</small>", unsafe_allow_html=True)
            st.file_uploader("Subir", label_visibility="collapsed")
            st.button("Analyze CT Scan", type="primary")
            st.markdown('</div>', unsafe_allow_html=True)
        with res:
            st.markdown('<div class="med-card" style="border-top:4px solid #E11D48">', unsafe_allow_html=True)
            st.markdown("<small>ANALYSIS RESULT</small> <span style='float:right; font-weight:bold;'>94.8%</span>", unsafe_allow_html=True)
            st.markdown("<h3 style='color:#E11D48; margin:10px 0;'>⚠️ Stroke detected</h3>", unsafe_allow_html=True)
            st.markdown("<p style='font-size:13px;'>Subtype: <b>Ischemic</b><br>Region: <b>Right MCA Territory</b></p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        st.image("https://via.placeholder.com/800x400.png?text=Visualización+de+Slices+CT+Scan", use_container_width=True)

    with col_side:
        right_sidebar()

# ─────────────────────────────────────────────
# PÁGINA 4: RESULTS (Fusion)
# ─────────────────────────────────────────────
def show_results():
    col_main, col_side = st.columns([8, 2.5])
    with col_main:
        st.markdown("## Final Assessment")
        st.markdown('<div class="med-card" style="border-left:8px solid #E11D48">', unsafe_allow_html=True)
        st.markdown("<h2 style='color:#E11D48; margin:0;'>86% High Likelihood of Acute Stroke</h2>", unsafe_allow_html=True)
        st.markdown("<p style='margin:10px 0;'>The model fusion analysis identifies a high probability of LVO based on clinical and imaging features.</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="med-card"><b>Tabular Model Risk</b><br>Patient Age (74): +12%<br><br><h2 style="color:#E11D48">82%</h2></div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="med-card"><b>CT Scan Analysis</b><br>ASPECTS Score: 6/10<br><br><h2 style="color:#E11D48">89%</h2></div>', unsafe_allow_html=True)
            
        st.info("🚨 **Immediate Recommendation:** Immediate neurological evaluation recommended. Consider emergent mechanical thrombectomy.")

    with col_side:
        right_sidebar()

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
render_navbar()
st.markdown("<br>", unsafe_allow_html=True)

if st.session_state.page == "Directory":
    show_directory()
elif st.session_state.page == "Data":
    show_data_entry()
elif st.session_state.page == "CT":
    show_ct_scan()
elif st.session_state.page == "Results":
    show_results()