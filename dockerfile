#  Usar una imagen oficial de Python ligera (basado en tu pyproject.toml que pide >=3.11)
FROM python:3.11-slim

RUN pip install uv

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar los archivos de configuración de dependencias primero

COPY pyproject.toml uv.lock ./

# Instalar las dependencias del proyecto usando uv
RUN uv sync --frozen

#  Copiar el resto del código del proyecto al contenedor
COPY . .

#  Exponer el puerto por defecto que usa Streamlit
EXPOSE 8501

#  Comando por defecto al encender el contenedor para ejecutar la aplicación Streamlit
CMD ["uv", "run", "streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]