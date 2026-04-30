import os
import uuid
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()
supabase: Client = create_client(os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY"))

def test_arquitectura_completa():
    try:
        # 1. GENERAR PACIENTE CON TODOS LOS DATOS (Fase 1)
        p_id = str(uuid.uuid4())
        
        paciente = {
            "patient_id": p_id,
            "gender": "Male",
            "age": 15.0,
            "hypertension": 1,
            "heart_disease": 0,
            "ever_married": "Yes",
            "work_type": "Private",
            "residence_type": "Urban",
            "avg_glucose_level": 110.5,
            "bmi": 28.4,
            "smoking_status": "formerly smoked",
            "prediction": 1,
            "probability": 0.85,
            "age_group": "children"
        }
        
        print(f"--- Registrando paciente {p_id} con historial completo ---")
        respuesta_sql = supabase.table("stroke_predictions").insert(paciente).execute()
        assert len(respuesta_sql.data) > 0, "No se guardó el paciente en la tabla SQL"
        
        # 2. REGISTRAR CONSTANTES VITALES (Nueva tabla)
        print(f"--- Registrando constantes vitales para {p_id} ---")
        vitals = {
            "patient_id": p_id,
            "patient_name": "Paciente de Prueba_Niño",
            "spo2": 98,
            "bpm": 75
        }
        respuesta_vitals = supabase.table("patient_vitals").insert(vitals).execute()
        assert len(respuesta_vitals.data) > 0, "No se guardaron las constantes vitales"
        print("✅ Tabla patient_vitals vinculada correctamente.")

        # 2. SUBIR IMAGEN AL BUCKET (Fase 2)
        ruta_imagen = "./assets/v2/prueba.png" 
        if os.path.exists(ruta_imagen):
            with open(ruta_imagen, "rb") as f:
                nombre_nube = f"{p_id}_scan.png"
                print(f"--- Subiendo imagen: {nombre_nube} ---")
                supabase.storage.from_("tomografias").upload(path=nombre_nube, file=f.read())
            
            # 3. GUARDAR EL LINK EN SQL (El Enlace)
            url = supabase.storage.from_("tomografias").get_public_url(nombre_nube)
            print(f"--- Vinculando imagen en DB ---")
            supabase.table("brain_scans").insert({"patient_id": p_id, "image_url": url}).execute()
            
            print("\n ¡SISTEMA INTEGRADO FUNCIONANDO!")
            print(f"Puedes ver la foto aquí: {url}")
            
            # --- TUS ASERCIONES PARA PYTEST ---
            assert url is not None, "El link no debería estar vacío"
            assert "supabase.co" in url, "Debería ser un link de Supabase"
            
        else:
            print("❌ No encontré el archivo 'prueba.png'.")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        assert False, f"El test falló con esta excepción: {e}" # Si hay error, que el test falle

# --- EL TRUCO PRO ---
# Si ejecutas "python test_sistema_completo.py", entrará por aquí.
# Si ejecutas "pytest", ignorará esto y solo testeará la función sin repetirla.
if __name__ == "__main__":
    test_arquitectura_completa()