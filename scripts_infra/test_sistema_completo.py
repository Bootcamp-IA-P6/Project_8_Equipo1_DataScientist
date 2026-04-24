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
            "age": 60.0,
            "hypertension": 1,
            "heart_disease": 0,
            "ever_married": "Yes",
            "work_type": "Private",
            "residence_type": "Urban",
            "avg_glucose_level": 110.5,
            "bmi": 28.4,
            "smoking_status": "formerly smoked",
            "prediction": 1,
            "probability": 0.85
        }
        
        print(f"--- Registrando paciente {p_id} con historial completo ---")
        supabase.table("stroke_predictions").insert(paciente).execute()

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
        else:
            print("❌ No encontré el archivo 'prueba.png' en la carpeta, pero la tabla de texto funcionó.")

    except Exception as e:
        print(f"\n❌ Error: {e}")

test_arquitectura_completa()