import os
import uuid
from dotenv import load_dotenv
from supabase import create_client, Client

# 1. Cargar las llaves secretas de tu archivo .env
load_dotenv()

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")

# Verificar que las claves se han cargado (sin imprimirlas por seguridad)
if not url or not key:
    print("❌ ERROR: No se encontraron SUPABASE_URL o SUPABASE_KEY en el archivo .env")
    exit()

# 2. Conectar con Supabase
print("Conectando a Supabase...")
supabase: Client = create_client(url, key)

# 3. Crear datos falsos simulando una predicción que vendría de Streamlit
id_paciente = str(uuid.uuid4()) # Genera un ID único para el paciente

datos_prueba = {
    "patient_id": id_paciente,
    "gender": "Female",
    "age": 45.0,
    "hypertension": 0,
    "heart_disease": 0,
    "ever_married": "Yes",
    "work_type": "Private",
    "residence_type": "Urban",
    "avg_glucose_level": 105.5,
    "bmi": 24.2,
    "smoking_status": "never smoked",
    "prediction": 0,          # 0 = No Ictus, 1 = Ictus
    "probability": 0.12       # 12% de probabilidad
}

# 4. Enviar los datos a la tabla 'stroke_predictions'
try:
    print("Enviando datos a la tabla 'stroke_predictions'...")
    # El comando .execute() es el que realmente lanza la información a la nube
    respuesta = supabase.table("stroke_predictions").insert(datos_prueba).execute()
    
    print("\n✅ ¡ÉXITO TOTAL!")
    print("Los datos se han guardado correctamente en tu base de datos.")
    print(f"ID del paciente insertado: {id_paciente}")
    
except Exception as e:
    print(f"\n❌ ERROR: No se pudo guardar. Detalle del error: {e}")