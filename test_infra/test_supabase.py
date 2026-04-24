import os
import uuid
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()
supabase: Client = create_client(os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY"))

def test_conexion_basica():
    try:
        # Creamos un paciente muy sencillo solo para ver si la puerta está abierta
        p_id = str(uuid.uuid4())
        paciente_simple = {
            "patient_id": p_id,
            "gender": "Female",
            "age": 45.0,
            "prediction": 0,
            "probability": 0.12
        }
        
        print(f"\n--- Testeando conexión básica a la tabla ---")
        respuesta = supabase.table("stroke_predictions").insert(paciente_simple).execute()
        
        # --- ASERCIONES PARA PYTEST ---
        assert respuesta.data is not None, "La respuesta de la base de datos es nula"
        assert len(respuesta.data) > 0, "No se insertó la fila correctamente"
        assert respuesta.data[0]["patient_id"] == p_id, "El ID de paciente guardado no coincide"
        
        print("✅ ¡Conexión básica exitosa!")
        
    except Exception as e:
        print(f"\n❌ Error en conexión básica: {e}")
        assert False, f"El test de conexión básica falló: {e}"


if __name__ == "__main__":
    test_conexion_basica()