#!/bin/bash
# start.sh - Script de inicio para Railway

# Obtener el puerto de Railway o usar 8000 por defecto
PORT=${PORT:-8000}

# Ejecutar la aplicación
exec uvicorn main:app --host 0.0.0.0 --port $PORT