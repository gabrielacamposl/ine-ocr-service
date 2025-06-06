# main.py - INE OCR Microservice
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import pytesseract
import re
import base64
import io
from PIL import Image
import logging
from typing import Optional, Dict, Tuple
import gc

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="INE OCR Microservice",
    description="Microservicio para extraer CIC e Identificador del Ciudadano de INE mexicana",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class INEProcessor:
    """Clase principal para procesar imágenes de INE y extraer datos"""
    
    def __init__(self):
        # Configurar pytesseract
        pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'  # Para Railway/Docker
        
        # Patrones regex para extraer datos específicos de INE
        self.patterns = {
            'cic': [
                r'IDMEX(\d{10})',  # Patrón principal CIC
                r'(\d{13})',       # CIC como secuencia de 13 dígitos
                r'CIC[:\s]*(\d+)', # CIC precedido por "CIC"
            ],
            'id_ciudadano': [
                r'(\d{13})',       # Identificador como secuencia de 13 dígitos
                r'(\d{10})',       # Identificador como secuencia de 10 dígitos
                r'ID[:\s]*(\d+)',  # ID precedido por "ID"
            ],
            'curp': [
                r'([A-Z]{4}\d{6}[HM][A-Z]{5}[A-Z\d]{2})',  # Patrón CURP estándar
                r'CURP[:\s]*([A-Z]{4}\d{6}[HM][A-Z]{5}[A-Z\d]{2})',
            ]
        }
    
    def aggressive_memory_cleanup(self):
        """Limpieza agresiva de memoria"""
        try:
            for _ in range(3):
                gc.collect()
            logger.info("Limpieza de memoria completada")
        except Exception as e:
            logger.error(f"Error en limpieza de memoria: {e}")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocesa la imagen para mejorar el OCR"""
        try:
            # Convertir a escala de grises
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Redimensionar si es muy grande (optimización para Railway)
            height, width = gray.shape
            if width > 1500 or height > 1000:
                scale = min(1500/width, 1000/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Aplicar filtro bilateral para reducir ruido manteniendo bordes
            denoised = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # Mejorar contraste usando CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)
            
            # Aplicar threshold adaptativo
            binary = cv2.adaptiveThreshold(
                enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # Operaciones morfológicas para limpiar texto
            kernel = np.ones((1,1), np.uint8)
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            return cleaned
            
        except Exception as e:
            logger.error(f"Error en preprocesamiento: {e}")
            return image
    
    def extract_text_regions(self, image: np.ndarray) -> list:
        """Extrae regiones de texto específicas de la INE"""
        try:
            height, width = image.shape
            
            # Definir regiones donde típicamente aparecen los datos en INE
            regions = [
                # Región principal donde aparece IDMEX y datos
                {"name": "main_data", "roi": (0, int(height*0.6), width, int(height*0.9))},
                # Región inferior donde aparecen códigos
                {"name": "bottom_codes", "roi": (0, int(height*0.8), width, height)},
                # Región central
                {"name": "center", "roi": (0, int(height*0.4), width, int(height*0.7))},
                # Toda la imagen como fallback
                {"name": "full", "roi": (0, 0, width, height)}
            ]
            
            extracted_regions = []
            
            for region in regions:
                x, y, x2, y2 = region["roi"]
                roi = image[y:y2, x:x2]
                
                if roi.size > 0:
                    extracted_regions.append({
                        "name": region["name"],
                        "image": roi,
                        "coordinates": region["roi"]
                    })
            
            return extracted_regions
            
        except Exception as e:
            logger.error(f"Error extrayendo regiones: {e}")
            return [{"name": "full", "image": image, "coordinates": (0, 0, width, height)}]
    
    def perform_ocr(self, image: np.ndarray) -> str:
        """Realiza OCR en la imagen usando diferentes configuraciones"""
        try:
            # Configuraciones de OCR para diferentes casos
            configs = [
                '--psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',  # Solo números y letras
                '--psm 7 -c tessedit_char_whitelist=0123456789',  # Solo números
                '--psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',  # Una palabra
                '--psm 6',  # Bloque de texto uniforme
                '--psm 4',  # Columna de texto
            ]
            
            results = []
            
            for config in configs:
                try:
                    text = pytesseract.image_to_string(image, config=config, lang='spa')
                    if text.strip():
                        results.append(text.strip())
                except:
                    continue
            
            # Combinar todos los resultados
            combined_text = '\n'.join(results)
            return combined_text
            
        except Exception as e:
            logger.error(f"Error en OCR: {e}")
            return ""
    
    def extract_data_with_patterns(self, text: str) -> Dict[str, Optional[str]]:
        """Extrae CIC, ID Ciudadano y CURP usando patrones regex"""
        extracted_data = {
            'cic': None,
            'id_ciudadano': None,
            'curp': None
        }
        
        # Limpiar texto
        cleaned_text = re.sub(r'\s+', ' ', text.upper().strip())
        
        # Extraer CIC
        for pattern in self.patterns['cic']:
            matches = re.findall(pattern, cleaned_text)
            if matches:
                # Buscar el CIC más probable (generalmente el más largo o el que sigue el patrón IDMEX)
                for match in matches:
                    if len(match) >= 10:  # CIC debe tener al menos 10 dígitos
                        extracted_data['cic'] = match
                        break
                if extracted_data['cic']:
                    break
        
        # Extraer Identificador del Ciudadano
        for pattern in self.patterns['id_ciudadano']:
            matches = re.findall(pattern, cleaned_text)
            if matches:
                for match in matches:
                    # Evitar duplicar el CIC como ID Ciudadano
                    if match != extracted_data['cic'] and len(match) >= 10:
                        extracted_data['id_ciudadano'] = match
                        break
                if extracted_data['id_ciudadano']:
                    break
        
        # Extraer CURP
        for pattern in self.patterns['curp']:
            matches = re.findall(pattern, cleaned_text)
            if matches:
                extracted_data['curp'] = matches[0]
                break
        
        return extracted_data
    
    def process_ine_image(self, image_file) -> Dict[str, Optional[str]]:
        """Procesa una imagen de INE y extrae los datos principales"""
        try:
            logger.info("Iniciando procesamiento de imagen INE")
            
            # Leer imagen desde archivo
            image_data = image_file.read()
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("No se pudo decodificar la imagen")
            
            # Preprocesar imagen
            processed_image = self.preprocess_image(image)
            
            # Extraer regiones de texto
            regions = self.extract_text_regions(processed_image)
            
            all_extracted_data = {
                'cic': None,
                'id_ciudadano': None,
                'curp': None
            }
            
            # Procesar cada región
            for region in regions:
                logger.info(f"Procesando región: {region['name']}")
                
                # Realizar OCR
                text = self.perform_ocr(region['image'])
                
                if text:
                    logger.info(f"Texto extraído de {region['name']}: {text[:100]}...")
                    
                    # Extraer datos con patrones
                    extracted_data = self.extract_data_with_patterns(text)
                    
                    # Actualizar datos si encontramos algo nuevo
                    for key, value in extracted_data.items():
                        if value and not all_extracted_data[key]:
                            all_extracted_data[key] = value
                            logger.info(f"Encontrado {key}: {value}")
            
            # Limpieza de memoria
            self.aggressive_memory_cleanup()
            
            return all_extracted_data
            
        except Exception as e:
            logger.error(f"Error procesando imagen INE: {e}")
            raise HTTPException(status_code=500, detail=f"Error procesando imagen: {str(e)}")

# Instancia del procesador
ine_processor = INEProcessor()

@app.get("/")
async def root():
    return {"message": "INE OCR Microservice", "status": "active"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "ine-ocr"}

@app.post("/extract-ine-data")
async def extract_ine_data(
    ine_front: UploadFile = File(..., description="Imagen frontal de la INE"),
    ine_back: UploadFile = File(None, description="Imagen trasera de la INE (opcional)")
):
    """
    Extrae CIC, Identificador del Ciudadano y CURP de las imágenes de INE
    """
    try:
        # Validar archivo frontal
        if not ine_front.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="El archivo frontal debe ser una imagen")
        
        # Procesar imagen frontal
        logger.info("Procesando imagen frontal de INE")
        front_data = ine_processor.process_ine_image(ine_front.file)
        
        result = {
            "success": True,
            "data": {
                "cic": front_data.get('cic'),
                "id_ciudadano": front_data.get('id_ciudadano'),
                "curp": front_data.get('curp')
            },
            "processed_images": ["front"]
        }
        
        # Si se proporcionó imagen trasera, procesarla también
        if ine_back and ine_back.content_type.startswith('image/'):
            logger.info("Procesando imagen trasera de INE")
            back_data = ine_processor.process_ine_image(ine_back.file)
            
            # Combinar datos (priorizar datos del frente, complementar con trasera)
            for key, value in back_data.items():
                if value and not result["data"][key]:
                    result["data"][key] = value
            
            result["processed_images"].append("back")
        
        # Verificar que se extrajeron datos mínimos
        if not result["data"]["cic"] and not result["data"]["id_ciudadano"]:
            return {
                "success": False,
                "error": "No se pudieron extraer los datos mínimos (CIC o ID Ciudadano)",
                "data": result["data"],
                "suggestion": "Verifica que la imagen sea clara y esté bien iluminada"
            }
        
        logger.info(f"Extracción exitosa: CIC={result['data']['cic']}, ID={result['data']['id_ciudadano']}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en endpoint extract-ine-data: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")

@app.post("/extract-ine-data-base64")
async def extract_ine_data_base64(request_data: dict):
    """
    Extrae datos de INE usando imágenes en formato base64
    """
    try:
        front_b64 = request_data.get('ine_front_b64')
        back_b64 = request_data.get('ine_back_b64')
        
        if not front_b64:
            raise HTTPException(status_code=400, detail="ine_front_b64 es requerido")
        
        # Decodificar imagen frontal
        try:
            front_data = base64.b64decode(front_b64)
            front_file = io.BytesIO(front_data)
            front_result = ine_processor.process_ine_image(front_file)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error decodificando imagen frontal: {str(e)}")
        
        result = {
            "success": True,
            "data": {
                "cic": front_result.get('cic'),
                "id_ciudadano": front_result.get('id_ciudadano'),
                "curp": front_result.get('curp')
            },
            "processed_images": ["front"]
        }
        
        # Procesar imagen trasera si está disponible
        if back_b64:
            try:
                back_data = base64.b64decode(back_b64)
                back_file = io.BytesIO(back_data)
                back_result = ine_processor.process_ine_image(back_file)
                
                # Combinar datos
                for key, value in back_result.items():
                    if value and not result["data"][key]:
                        result["data"][key] = value
                
                result["processed_images"].append("back")
            except Exception as e:
                logger.warning(f"Error procesando imagen trasera: {e}")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en endpoint base64: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)