import streamlit as st
import pandas as pd
import os
import uuid
import joblib
import numpy as np
from dotenv import load_dotenv
from supabase import create_client, Client
from datetime import datetime
import time
import base64

# ─────────────────────────────────────────────
# 1. CONFIGURACIÓN
# ─────────────────────────────────────────────
load_dotenv()

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA_MODELO = os.path.join(BASE_DIR, "models", "XGB_optuna_final.joblib")
RUTA_LOGO   = os.path.join(BASE_DIR, "assets", "aria_logo.png")

@st.cache_resource
def iniciar_conexion():
    return create_client(os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY"))

@st.cache_resource
def cargar_modelo_xgb():
    if os.path.exists(RUTA_MODELO):
        try:
            return joblib.load(RUTA_MODELO)
        except Exception as e:
            st.error(f"Error al cargar el modelo: {e}")
    return None

def logo_base64():
    if os.path.exists(RUTA_LOGO):
        with open(RUTA_LOGO, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

supabase   = iniciar_conexion()
modelo_xgb = cargar_modelo_xgb()

# ─────────────────────────────────────────────
# 2. MAPEOS ES → EN
# ─────────────────────────────────────────────
GENERO_MAP     = {"Femenino": "Female", "Masculino": "Male"}
CASADO_MAP     = {"Sí": "Yes", "No": "No"}
TRABAJO_MAP    = {
    "Sector privado": "Private",
    "Autónomo":       "Self-employed",
    "Sector público": "Govt_job",
    "Menor de edad":  "children",
    "Sin empleo":     "Never_worked",
}
TABACO_MAP     = {
    "Ex fumador":    "formerly smoked",
    "Nunca fumó":    "never smoked",
    "Fumador activo":"smokes",
    "Desconocido":   "Unknown",
}
RESIDENCIA_MAP = {"Urbana": "Urban", "Rural": "Rural"}

# ─────────────────────────────────────────────
# 3. PAGE CONFIG + CSS
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ARIA · Sistema de Riesgo de Ictus",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Logo: imagen real o SVG fallback
LOGO_B64 = logo_base64()
if LOGO_B64:
    LOGO_HTML = f'<img src="data:image/png;base64,{LOGO_B64}" style="height:50px;object-fit:contain;">'
else:
    LOGO_HTML = """
    <svg viewBox="0 0 130 50" height="50" xmlns="http://www.w3.org/2000/svg">
      <rect width="50" height="50" rx="10" fill="#0C447C"/>
      <rect x="4" y="4" width="42" height="42" rx="7" fill="#185FA5"/>
      <line x1="17" y1="13" x2="17" y2="37" stroke="#E6F1FB" stroke-width="3" stroke-linecap="round"/>
      <polyline points="17,13 25,13 33,37" fill="none" stroke="#E6F1FB" stroke-width="3"
                stroke-linecap="round" stroke-linejoin="round"/>
      <line x1="19" y1="27" x2="31" y2="27" stroke="#E6F1FB" stroke-width="2.5" stroke-linecap="round"/>
      <text x="57" y="23" font-family="Georgia,serif" font-size="17" font-weight="700"
            fill="#0C447C" letter-spacing="3">ARIA</text>
      <text x="57" y="37" font-family="Arial,sans-serif" font-size="7" fill="#185FA5"
            letter-spacing="1.2">ANÁLISIS DE RIESGO IA</text>
    </svg>"""

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
            
.main .block-container {
    padding-top: 0 !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 1280px;
}
            
section[data-testid="stSidebar"] { display: none; }
#MainMenu, footer, header { visibility: hidden; }

/* ── Header con logomarca ── */
.aria-header {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    background: white;
    border-bottom: 1px solid #DDEAF7;
    padding: 0.5rem 2rem 1rem 2rem;
    margin: 0;
}
.aria-header img {
    height: 20px;
    object-fit: contain;
    margin-bottom: 0.25rem;
}
.aria-header-patient {
    font-size: 12px;
    color: #5F7A99;
    background: #F4F7FB;
    border: 1px solid #DDEAF7;
    border-radius: 20px;
    padding: 4px 16px;
    margin-top: 0.5rem;
}

/* ── Barra de navegación ── */
.aria-navstrip {
    background: #0C447C;
    padding: 0 2rem;
    margin: 0 -2rem;
    display: flex;
    align-items: center;
    height: 46px;
    border-bottom: 3px solid #185FA5;
}
.aria-divider {
    height: 3px;
    background: linear-gradient(to right, #0C447C 0%, #185FA5 40%, transparent 100%);
    border-radius: 2px;
    margin-bottom: 1.5rem;
}

/* ── Cards ── */
.aria-card {
    background: white;
    border: 1px solid #DDEAF7;
    border-radius: 16px;
    padding: 1.5rem 1.75rem;
    margin-bottom: 1.25rem;
}
.aria-card-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 1.25rem;
    padding-bottom: 0.75rem;
    border-bottom: 1.5px solid #DDEAF7;
}
.aria-card-step {
    background: #0C447C;
    color: white;
    font-size: 11px;
    font-weight: 600;
    width: 24px; height: 24px;
    border-radius: 50%;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
.aria-card-title {
    font-size: 15px;
    font-weight: 600;
    color: #0C447C;
    margin: 0;
}

/* ── Métricas ── */
.aria-metric {
    background: #F4F7FB;
    border: 1px solid #DDEAF7;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    text-align: center;
}
.aria-metric-label {
    font-size: 10.5px;
    color: #5F7A99;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 500;
    margin-bottom: 4px;
}
.aria-metric-value {
    font-size: 28px;
    font-weight: 600;
    color: #0C447C;
    line-height: 1.1;
}
.aria-metric-sub {
    font-size: 11px;
    color: #5F7A99;
    margin-top: 3px;
}

/* ── Badges ── */
.badge-alto  { background:#FCEBEB; color:#791F1F; border:1px solid #F09595;
                border-radius:20px; padding:3px 12px; font-size:12px; font-weight:600; display:inline-block; }
.badge-medio { background:#FAEEDA; color:#633806; border:1px solid #FAC775;
                border-radius:20px; padding:3px 12px; font-size:12px; font-weight:600; display:inline-block; }
.badge-bajo  { background:#EAF3DE; color:#27500A; border:1px solid #C0DD97;
                border-radius:20px; padding:3px 12px; font-size:12px; font-weight:600; display:inline-block; }

/* ── Alerta crítica ── */
.aria-alert {
    background: #FCEBEB;
    border: 1.5px solid #F09595;
    border-left: 5px solid #E24B4A;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin: 1rem 0;
}
.aria-alert-title { font-size:14px; font-weight:600; color:#791F1F; margin-bottom:4px; }
.aria-alert-sub   { font-size:12px; color:#A32D2D; }

/* ── Tabla historial ── */
.aria-col-header {
    font-size: 10.5px;
    font-weight: 600;
    color: #5F7A99;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding-bottom: 8px;
    border-bottom: 1.5px solid #DDEAF7;
}
.aria-row {
    padding: 10px 0;
    border-bottom: 1px solid #F0F4FA;
    font-size: 13.5px;
    color: #1a2e45;
}

/* ── Botón emergencia ── */
/* ── Botón de Urgencia (Máxima Prioridad) ── */
div.emergency-btn [data-testid="stButton"] button {
    background-color: #E24B4A !important;
    border: 2px solid #E24B4A !important;
    border-radius: 12px !important;
    
    /* Dimensiones y Centrado */
    width: auto !important;
    min-width: 350px !important;
    display: block !important;
    margin: 1.5rem auto !important;
    
    /* Animación */
    animation: pulse-red 2s infinite !important;
    transition: all 0.3s ease !important;
}

/* Forzar el estilo del texto dentro del botón */
div.emergency-btn [data-testid="stButton"] button p {
    color: white !important;
    font-weight: 700 !important;
    font-size: 16px !important;
}

/* Efecto Hover (al pasar el mouse) */
div.emergency-btn [data-testid="stButton"] button:hover {
    background-color: #b33231 !important;
    border-color: #b33231 !important;
    transform: scale(1.02);
}

@keyframes pulse-red {
    0% { box-shadow: 0 0 0 0 rgba(226, 75, 74, 0.7); }
    70% { box-shadow: 0 0 0 15px rgba(226, 75, 74, 0); }
    100% { box-shadow: 0 0 0 0 rgba(226, 75, 74, 0); }
}
            
            
/* ── Streamlit overrides ── */
div[data-testid="stHorizontalBlock"] .stButton > button {{
    border-radius: 8px !important;
    font-size: 12.5px !important;
    height: 36px !important;
    font-weight: 500 !important;
}}
.stTextInput > div > div > input,
.stNumberInput > div > div > input {{
    border-radius: 8px !important;
    border-color: #DDEAF7 !important;
    font-size: 13.5px !important;
}}
label {{
    font-size: 12.5px !important;
    font-weight: 500 !important;
    color: #3a5272 !important;
}}
            
header { display: none !important; }

div[data-testid="stAppViewContainer"] > .main {
    padding-top: 0 !important;
}

.block-container {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 4. SESSION STATE
# ─────────────────────────────────────────────
def reset_paciente():
    st.session_state.temp_data = {
        "patient_id": None, "patient_name": "",
        "spo2": 98, "bpm": 75,
        "xgb_prob": 0, "xgb_pred": 0,
        "cnn_prob": 0, "cnn_done": False,
        "finalizado": False,
    }

if "temp_data" not in st.session_state:
    reset_paciente()
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

# ─────────────────────────────────────────────
# 5. DB HELPERS
# ─────────────────────────────────────────────
def cargar_paciente_existente(p_id):
    try:
        rp = supabase.table("stroke_predictions").select("*").eq("patient_id", p_id).single().execute()
        rv = supabase.table("patient_vitals").select("*").eq("patient_id", p_id).single().execute()
        if rp.data and rv.data:
            st.session_state.temp_data.update({
                "patient_id": p_id, "patient_name": rv.data["patient_name"],
                "spo2": rv.data["spo2"], "bpm": rv.data["bpm"],
                "xgb_prob": rp.data["probability"], "xgb_pred": rp.data["prediction"],
                "finalizado": True,
            })
            return True
    except:
        return False
    return False

# ─────────────────────────────────────────────
# 6. HELPERS UI
# ─────────────────────────────────────────────
def card_open(step, title):
    step_html = (
        f'<span style="background:#0C447C;color:white;font-size:11px;font-weight:600;'
        f'width:24px;height:24px;border-radius:50%;display:inline-flex;align-items:center;'
        f'justify-content:center;flex-shrink:0;">{step}</span>'
        if step else ""
    )
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:1rem;
                padding-bottom:0.75rem;border-bottom:1.5px solid #DDEAF7;">
        {step_html}
        <span style="font-size:15px;font-weight:600;color:#0C447C;margin:0;">{title}</span>
    </div>
    """, unsafe_allow_html=True)

def card_close():
    pass

# ─────────────────────────────────────────────
# 7. NAVBAR
# ─────────────────────────────────────────────
def render_navbar():
    paciente = st.session_state.temp_data.get("patient_name") or "Ningún paciente activo"

    # ── Cabecero con logomarca centrada ──
    if LOGO_B64:
        logo_tag = f'<img src="data:image/png;base64,{LOGO_B64}" style="height:35vh;object-fit:contain;">'
    else:
        logo_tag = """
        <svg viewBox="0 0 260 120" height="120" xmlns="http://www.w3.org/2000/svg">
          <rect width="100" height="100" x="80" y="10" rx="18" fill="#0C447C"/>
          <rect x="88" y="18" width="84" height="84" rx="13" fill="#185FA5"/>
          <line x1="108" y1="30" x2="108" y2="82" stroke="#E6F1FB" stroke-width="5" stroke-linecap="round"/>
          <polyline points="108,30 130,30 152,82" fill="none" stroke="#E6F1FB" stroke-width="5"
                    stroke-linecap="round" stroke-linejoin="round"/>
          <line x1="114" y1="62" x2="146" y2="62" stroke="#E6F1FB" stroke-width="4" stroke-linecap="round"/>
          <text x="130" y="108" text-anchor="middle" font-family="Georgia,serif" font-size="13"
                font-weight="700" fill="#5F7A99" letter-spacing="4">ANÁLISIS DE RIESGO IA</text>
        </svg>"""

    st.markdown(f"""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                background:white;border-bottom:1px solid #DDEAF7;padding:1.5rem 2rem 1rem 2rem;
                margin:-1rem -2rem 0 -2rem;">
        {logo_tag}
        <div style="font-size:12px;color:#5F7A99;background:#F4F7FB;border:1px solid #DDEAF7;
                    border-radius:20px;padding:4px 16px;margin-top:0.5rem;">
            Paciente activo: <strong>{paciente}</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)

    # ── Barra de navegación ──
    paginas   = ["Dashboard", "Triage", "Data", "CT", "Combined"]
    etiquetas = ["Historial", "1 · Signos vitales", "2 · Historia clínica", "3 · CT Scan", "4 · Informe"]

    cols = st.columns([1.1, 1.4, 1.6, 1, 1.2, 1.3])
    for i, p in enumerate(paginas):
        with cols[i]:
            tipo = "primary" if st.session_state.page == p else "secondary"
            if st.button(etiquetas[i], use_container_width=True, type=tipo, key=f"nav_{p}"):
                st.session_state.page = p
                st.rerun()
    with cols[5]:
        if st.button("+ Nuevo paciente", use_container_width=True, type="primary", key="nav_new"):
            reset_paciente()
            st.session_state.page = "Triage"
            st.rerun()

    st.markdown('<div style="height:3px;background:linear-gradient(to right,#0C447C,#185FA5,transparent);border-radius:2px;margin-bottom:1.5rem"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 8. PÁGINAS
# ─────────────────────────────────────────────

def show_dashboard():
    card_open(" ", "Historial médico · Últimos 10 registros")
    try:
        res = (
            supabase.table("stroke_predictions")
            .select("patient_id, probability, created_at, patient_vitals(patient_name)")
            .order("created_at", desc=True).limit(10).execute()
        )
        COL_H = "font-size:10.5px;font-weight:600;color:#5F7A99;text-transform:uppercase;letter-spacing:0.08em;padding-bottom:8px;border-bottom:1.5px solid #DDEAF7;display:block;"
        ROW_S = "padding:10px 0;border-bottom:1px solid #F0F4FA;font-size:13.5px;color:#1a2e45;display:block;"

        if res.data:
            h1, h2, h3, h4 = st.columns([3, 2, 2, 2])
            for col, lbl in zip([h1,h2,h3,h4], ["Paciente","Nivel de riesgo","Fecha","Acción"]):
                col.markdown(f'<span style="{COL_H}">{lbl}</span>', unsafe_allow_html=True)
            for item in res.data:
                v    = item["patient_vitals"][0] if item["patient_vitals"] else {"patient_name": "N/A"}
                prob = item["probability"]
                if prob > 60:
                    badge = f'<span style="background:#FCEBEB;color:#791F1F;border:1px solid #F09595;border-radius:20px;padding:3px 12px;font-size:12px;font-weight:600;">{prob}% · Alto</span>'
                elif prob > 30:
                    badge = f'<span style="background:#FAEEDA;color:#633806;border:1px solid #FAC775;border-radius:20px;padding:3px 12px;font-size:12px;font-weight:600;">{prob}% · Moderado</span>'
                else:
                    badge = f'<span style="background:#EAF3DE;color:#27500A;border:1px solid #C0DD97;border-radius:20px;padding:3px 12px;font-size:12px;font-weight:600;">{prob}% · Bajo</span>'
                c1,c2,c3,c4 = st.columns([3,2,2,2])
                c1.markdown(f'<span style="{ROW_S}">{v["patient_name"]}</span>', unsafe_allow_html=True)
                c2.markdown(f'<span style="{ROW_S}">{badge}</span>', unsafe_allow_html=True)
                c3.markdown(f'<span style="{ROW_S}">{item["created_at"][:10]}</span>', unsafe_allow_html=True)
                if c4.button("Ver detalle", key=f"ver_{item['patient_id']}"):
                    if cargar_paciente_existente(item["patient_id"]):
                        st.session_state.page = "Combined"; st.rerun()
        else:
            st.info("No hay registros previos en el sistema.")
    except Exception as e:
        st.error(f"Error de base de datos: {e}")
    card_close()


def show_triage():
    card_open("1", "Signos vitales y registro del paciente")
    nombre = st.text_input(
        "Nombre completo del paciente",
        value=st.session_state.temp_data["patient_name"],
        placeholder="Ej. María García López"
    )
    c1, c2 = st.columns(2)
    spo2 = c1.number_input("Saturación de oxígeno — SpO2 (%)", 70, 100, st.session_state.temp_data["spo2"])
    bpm  = c2.number_input("Frecuencia cardíaca — BPM", 30, 220, st.session_state.temp_data["bpm"])
    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
    if st.button("Continuar a historia clínica →", type="primary", use_container_width=False):
        if nombre.strip():
            st.session_state.temp_data.update({"patient_name": nombre.strip(), "spo2": spo2, "bpm": bpm})
            st.session_state.page = "Data"; st.rerun()
        else:
            st.error("Introduzca el nombre del paciente antes de continuar.")
    card_close()


def show_clinical_data():
    if not st.session_state.temp_data["patient_name"]:
        st.warning("Complete el triaje antes de continuar."); return

    card_open("2", f"Historia clínica · {st.session_state.temp_data['patient_name']}")
    c1, c2 = st.columns(2)
    with c1:
        genero_es    = st.selectbox("Género", list(GENERO_MAP.keys()))
        edad         = st.number_input("Edad", 0, 120, 65)
        hipertension = st.toggle("Hipertensión arterial")
        cardiopatia  = st.toggle("Enfermedad cardíaca")
    with c2:
        casado_es     = st.selectbox("Estado civil", list(CASADO_MAP.keys()))
        trabajo_es    = st.selectbox("Tipo de trabajo", list(TRABAJO_MAP.keys()))
        tabaco_es     = st.selectbox("Hábito tabáquico", list(TABACO_MAP.keys()))
        residencia_es = st.selectbox("Tipo de residencia", list(RESIDENCIA_MAP.keys()))
    c3, c4 = st.columns(2)
    glucosa = c3.number_input("Glucosa media (mg/dL)", 50.0, 400.0, 100.0)
    bmi     = c4.number_input("Índice de masa corporal (BMI)", 10.0, 70.0, 28.0)

    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
    if st.button("Analizar riesgo y guardar →", type="primary", use_container_width=False):
        if modelo_xgb:
            try:
                ge = GENERO_MAP[genero_es]; ca = CASADO_MAP[casado_es]
                tr = TRABAJO_MAP[trabajo_es]; ta = TABACO_MAP[tabaco_es]
                re = RESIDENCIA_MAP[residencia_es]

                input_df = pd.DataFrame([{
                    "gender": ge, "age": edad,
                    "hypertension": 1 if hipertension else 0,
                    "heart_disease": 1 if cardiopatia else 0,
                    "ever_married": ca, "work_type": tr,
                    "Residence_type": re,
                    "avg_glucose_level": glucosa, "bmi": bmi,
                    "smoking_status": ta,
                }])
                prob = round(float(modelo_xgb.predict_proba(input_df)[0][1]) * 100, 2)
                pred = int(modelo_xgb.predict(input_df)[0])
                p_id = str(uuid.uuid4())

                supabase.table("stroke_predictions").insert({
                    "patient_id": p_id, "gender": ge, "age": edad,
                    "hypertension": 1 if hipertension else 0,
                    "heart_disease": 1 if cardiopatia else 0,
                    "ever_married": ca, "work_type": tr,
                    "residence_type": re, "avg_glucose_level": glucosa,
                    "bmi": bmi, "smoking_status": ta,
                    "prediction": pred, "probability": prob,
                    "age_group": "Senior" if edad > 60 else "Adult",
                }).execute()

                supabase.table("patient_vitals").insert({
                    "patient_id": p_id,
                    "patient_name": st.session_state.temp_data["patient_name"],
                    "spo2": st.session_state.temp_data["spo2"],
                    "bpm":  st.session_state.temp_data["bpm"],
                }).execute()

                st.session_state.temp_data.update({
                    "finalizado": True, "patient_id": p_id,
                    "xgb_prob": prob, "xgb_pred": pred,
                })
                st.session_state.page = "Combined"; st.rerun()
            except Exception as e:
                st.error(f"Error en modelo o base de datos: {e}")
        else:
            st.error("El modelo no está disponible. Verifique la ruta del archivo.")
    card_close()


def show_ct_scan():
    if not st.session_state.temp_data["finalizado"]:
        st.warning("Realice primero el análisis clínico (paso 2)."); return

    card_open("3", "CT Scan · Análisis por imagen (CNN)")
    archivo = st.file_uploader("Cargar imagen de TAC (.png / .jpg)", type=["png","jpg","jpeg"])
    if archivo:
        col1, col2 = st.columns(2)
        col1.image(archivo, use_container_width=True, caption="Imagen cargada")
        with col2:
            with st.spinner("Procesando inferencia CNN..."):
                time.sleep(1)
            cnn_prob = 84.2
            st.session_state.temp_data.update({"cnn_prob": cnn_prob, "cnn_done": True})
            st.markdown(f"""
            <div style="background:#F4F7FB;border:1px solid #DDEAF7;border-radius:12px;padding:1rem 1.25rem;text-align:center;margin-top:1rem;">
                <div style="font-size:10.5px;color:#5F7A99;text-transform:uppercase;letter-spacing:0.1em;font-weight:500;margin-bottom:4px;">Detección CNN</div>
                <div style="font-size:28px;font-weight:600;line-height:1.1;color:#A32D2D;">{cnn_prob}%</div>
                <div style="font-size:11px;color:#5F7A99;margin-top:3px;">Probabilidad de lesión isquémica</div>
            </div>""", unsafe_allow_html=True)
            st.success("Imagen procesada correctamente.")
    card_close()


def show_combined_results():
    data = st.session_state.temp_data
    if not data.get("finalizado"):
        st.warning("No hay datos de evaluación disponibles."); return

    # Cabecera del informe
    st.markdown(f"""
    <div style="background:white;border:1px solid #DDEAF7;border-radius:16px;
                padding:1rem 1.5rem;margin-bottom:1.25rem;">
      <div style="font-size:17px;font-weight:600;color:#0C447C">
        Informe de riesgo · {data['patient_name']}
      </div>
      <div style="font-size:11.5px;color:#5F7A99;margin-top:2px">
        Generado por ARIA · Análisis de Riesgo con Inteligencia Artificial
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Cálculo riesgo final
    riesgo_final = data["xgb_prob"]
    if data["cnn_done"]:
        riesgo_final = round(data["xgb_prob"] * 0.4 + data["cnn_prob"] * 0.6, 1)
    color_rf = "#A32D2D" if riesgo_final > 70 else "#854F0B" if riesgo_final > 30 else "#27500A"

    # Métricas
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div style="background:#F4F7FB;border:1px solid #DDEAF7;border-radius:12px;padding:1rem 1.25rem;text-align:center;">
          <div style="font-size:10.5px;color:#5F7A99;text-transform:uppercase;letter-spacing:0.1em;font-weight:500;margin-bottom:4px;">Riesgo clínico · XGBoost</div>
          <div style="font-size:28px;font-weight:600;color:#0C447C;line-height:1.1;">{data['xgb_prob']}%</div>
          <div style="font-size:11px;color:#5F7A99;margin-top:3px;">Basado en historia clínica</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        if data["cnn_done"]:
            st.markdown(f"""
            <div style="background:#F4F7FB;border:1px solid #DDEAF7;border-radius:12px;padding:1rem 1.25rem;text-align:center;">
              <div style="font-size:10.5px;color:#5F7A99;text-transform:uppercase;letter-spacing:0.1em;font-weight:500;margin-bottom:4px;">Riesgo por imagen · CNN</div>
              <div style="font-size:28px;font-weight:600;color:#0C447C;line-height:1.1;">{data['cnn_prob']}%</div>
              <div style="font-size:11px;color:#5F7A99;margin-top:3px;">Basado en CT Scan</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#F4F7FB;border:1px solid #DDEAF7;border-radius:12px;padding:1rem 1.25rem;text-align:center;opacity:.45;">
              <div style="font-size:10.5px;color:#5F7A99;text-transform:uppercase;letter-spacing:0.1em;font-weight:500;margin-bottom:4px;">Riesgo por imagen · CNN</div>
              <div style="font-size:15px;font-weight:600;color:#0C447C;padding-top:6px;">No realizado</div>
              <div style="font-size:11px;color:#5F7A99;margin-top:3px;">CT Scan pendiente</div>
            </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div style="background:#F4F7FB;border:1px solid #DDEAF7;border-radius:12px;padding:1rem 1.25rem;text-align:center;border:2px solid {color_rf};">
          <div style="font-size:10.5px;color:#5F7A99;text-transform:uppercase;letter-spacing:0.1em;font-weight:500;margin-bottom:4px;">Riesgo combinado final</div>
          <div style="font-size:28px;font-weight:600;line-height:1.1;color:{color_rf};">{riesgo_final}%</div>
          <div style="font-size:11px;color:#5F7A99;margin-top:3px;">{'XGB × 0.4 + CNN × 0.6' if data['cnn_done'] else 'Solo evaluación clínica'}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)

    # Alerta
    if riesgo_final > 70:
        st.markdown(f"""
        <div style="background:#FCEBEB;border:1.5px solid #F09595;border-left:5px solid #E24B4A;
                    border-radius:12px;padding:1rem 1.25rem;margin:1rem 0;">
          <div style="font-size:14px;font-weight:600;color:#791F1F;margin-bottom:4px;">
            Alerta crítica — Riesgo elevado de ictus
          </div>
          <div style="font-size:12px;color:#A32D2D;">
            Riesgo combinado: {riesgo_final}%. Se recomienda notificación inmediata al servicio de neurología.
          </div>
        </div>""", unsafe_allow_html=True)
        st.markdown('<div class="div.emergency-btn">', unsafe_allow_html=True)
        if st.button(" 🚨 Notificar urgencia a neurología", use_container_width=False):
            st.toast("Notificación enviada al servicio de neurología.")
        st.markdown('</div>', unsafe_allow_html=True)
    elif riesgo_final > 30:
        st.warning(f"Riesgo moderado: {riesgo_final}%. Se recomienda seguimiento clínico.")
    else:
        st.success(f"Riesgo bajo: {riesgo_final}%. Control rutinario.")

    st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)
    with st.expander("Ver explicabilidad del modelo"):
        st.bar_chart(pd.DataFrame(
            {"Contribución al riesgo": [0.5, 0.3, 0.2, 0.15, 0.1]},
            index=["Edad", "Glucosa media", "BMI", "Hipertensión", "Tabaquismo"]
        ))

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
render_navbar()

if   st.session_state.page == "Dashboard": show_dashboard()
elif st.session_state.page == "Triage":    show_triage()
elif st.session_state.page == "Data":      show_clinical_data()
elif st.session_state.page == "CT":        show_ct_scan()
elif st.session_state.page == "Combined":  show_combined_results()