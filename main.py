from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import uvicorn
import numpy as np
import re
import pytesseract
from PIL import Image
import logging
from typing import Dict, Optional
import gc

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="INE OCR Service",
    description="Microservicio para extraer CIC e Identificador del Ciudadano de INE mexicana",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especifica los dominios permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configurar Tesseract (ajustar según el entorno)
# En Railway/Docker puede estar en /usr/bin/tesseract
try:
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
except:
    pass  # Usar la configuración por defecto

class INEOCRProcessor:
    def __init__(self):
        # Patrones regex para extraer datos de INE
        self.cic_patterns = [
            r'(?:CIC\s*[:\-]?\s*)?([A-Z]{2,6}\d{11,15})',
            r'(IDMEX\d{11,15})',
            r'([A-Z]{4,6}\d{10,15})',
            r'(\d{13,18})',  # Patrón numérico largo
        ]
        
        self.id_ciudadano_patterns = [
            r'<<(\d{11,15})',  # Después de <<
            r'(\d{12,15})',    # Números de 12-15 dígitos
            r'<(\d{11,15})',   # Después de <
        ]
        
    def aggressive_memory_cleanup(self):
        """Limpieza agresiva de memoria"""
        try:
            for _ in range(3):
                gc.collect()
            logger.info("Limpieza de memoria completada")
        except Exception as e:
            logger.error(f"Error en limpieza de memoria: {e}")

    def preprocess_image_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """Preprocesa la imagen para mejorar el OCR"""
        try:
            # Convertir a escala de grises
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Aplicar filtro bilateral para reducir ruido manteniendo bordes
            denoised = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # Mejorar contraste usando CLAHE
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)
            
            # Aplicar threshold adaptativo
            thresh = cv2.adaptiveThreshold(
                enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Operaciones morfológicas para limpiar
            kernel = np.ones((1,1), np.uint8)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
            
            return cleaned
            
        except Exception as e:
            logger.error(f"Error en preprocesamiento: {e}")
            return image

    def extract_text_from_image(self, image: np.ndarray, config: str = None) -> str:
        """Extrae texto de la imagen usando Tesseract"""
        try:
            if config is None:
                # Configuración optimizada para INE
                config = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<>'
            
            # Convertir numpy array a PIL Image
            pil_image = Image.fromarray(image)
            
            # Extraer texto
            text = pytesseract.image_to_string(pil_image, config=config, lang='eng')
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error en extracción de texto: {e}")
            return ""

    def extract_cic_from_text(self, text: str) -> Optional[str]:
        """Extrae el CIC del texto usando patrones regex"""
        try:
            # Limpiar texto
            cleaned_text = re.sub(r'\s+', '', text.upper())
            
            logger.info(f"Buscando CIC en texto: {cleaned_text[:200]}...")
            
            for pattern in self.cic_patterns:
                matches = re.findall(pattern, cleaned_text)
                if matches:
                    for match in matches:
                        # Validar que tenga la longitud esperada
                        if len(match) >= 11:
                            logger.info(f"CIC encontrado: {match}")
                            return match
            
            # Intentar extraer de líneas específicas
            lines = text.strip().split('\n')
            for line in lines:
                line_clean = re.sub(r'\s+', '', line.upper())
                if 'IDMEX' in line_clean or len(line_clean) > 15:
                    for pattern in self.cic_patterns:
                        matches = re.findall(pattern, line_clean)
                        if matches:
                            for match in matches:
                                if len(match) >= 11:
                                    return match
            
            return None
            
        except Exception as e:
            logger.error(f"Error extrayendo CIC: {e}")
            return None

    def extract_id_ciudadano_from_text(self, text: str) -> Optional[str]:
        """Extrae el Identificador del Ciudadano del texto"""
        try:
            # Limpiar texto
            cleaned_text = re.sub(r'\s+', '', text.upper())
            
            logger.info(f"Buscando ID Ciudadano en texto...")
            
            for pattern in self.id_ciudadano_patterns:
                matches = re.findall(pattern, cleaned_text)
                if matches:
                    for match in matches:
                        # Validar longitud
                        if 11 <= len(match) <= 15:
                            logger.info(f"ID Ciudadano encontrado: {match}")
                            return match
            
            # Buscar números largos en líneas específicas
            lines = text.strip().split('\n')
            for line in lines:
                line_clean = re.sub(r'\s+', '', line.upper())
                # Buscar números de longitud específica
                numbers = re.findall(r'\d{11,15}', line_clean)
                if numbers:
                    for num in numbers:
                        if 11 <= len(num) <= 15:
                            return num
            
            return None
            
        except Exception as e:
            logger.error(f"Error extrayendo ID Ciudadano: {e}")
            return None

    def process_ine_image(self, image_data: bytes) -> Dict[str, Optional[str]]:
        """Procesa una imagen de INE y extrae CIC e ID Ciudadano"""
        try:
            # Convertir bytes a numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("No se pudo decodificar la imagen")
            
            # Redimensionar si es muy grande
            height, width = image.shape[:2]
            if width > 2000 or height > 2000:
                scale = min(2000/width, 2000/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Preprocesar imagen
            processed_image = self.preprocess_image_for_ocr(image)
            
            # Extraer texto con múltiples configuraciones
            configs = [
                '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<>',
                '--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<>',
                '--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<>',
                '--oem 3 --psm 6',
            ]
            
            all_text = ""
            for config in configs:
                text = self.extract_text_from_image(processed_image, config)
                all_text += text + "\n"
            
            logger.info(f"Texto extraído: {all_text[:500]}...")
            
            # Extraer CIC e ID Ciudadano
            cic = self.extract_cic_from_text(all_text)
            id_ciudadano = self.extract_id_ciudadano_from_text(all_text)
            
            # Limpieza de memoria
            self.aggressive_memory_cleanup()
            
            return {
                "cic": cic,
                "id_ciudadano": id_ciudadano,
                "text_extracted": all_text[:1000] if all_text else ""  # Para debug
            }
            
        except Exception as e:
            logger.error(f"Error procesando imagen INE: {e}")
            self.aggressive_memory_cleanup()
            raise HTTPException(status_code=400, detail=f"Error procesando imagen: {str(e)}")

# Instancia del procesador
ocr_processor = INEOCRProcessor()

@app.get("/")
async def root():
    return {
        "message": "INE OCR Service",
        "version": "1.0.0",
        "endpoints": {
            "extract": "POST /extract - Extrae CIC e ID Ciudadano de imágenes INE",
            "health": "GET /health - Verificar estado del servicio"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "ine-ocr"}

@app.post("/extract")
async def extract_ine_data(
    ine_front: UploadFile = File(...),
    ine_back: UploadFile = File(...)
):
    """
    Extrae CIC e Identificador del Ciudadano de las imágenes frontal y trasera de la INE
    """
    try:
        logger.info("=== INICIANDO EXTRACCIÓN OCR DE INE ===")
        
        # Validar archivos
        if not ine_front.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="El archivo frontal debe ser una imagen")
        
        if not ine_back.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="El archivo trasero debe ser una imagen")
        
        # Leer contenido de las imágenes
        front_content = await ine_front.read()
        back_content = await ine_back.read()
        
        # Procesar imagen frontal
        logger.info("Procesando imagen frontal...")
        front_data = ocr_processor.process_ine_image(front_content)
        
        # Procesar imagen trasera
        logger.info("Procesando imagen trasera...")
        back_data = ocr_processor.process_ine_image(back_content)
        
        # Combinar resultados (preferir datos del frente, usar trasera como backup)
        cic = front_data.get("cic") or back_data.get("cic")
        id_ciudadano = front_data.get("id_ciudadano") or back_data.get("id_ciudadano")
        
        # Validar que se extrajeron los datos esenciales
        if not cic or not id_ciudadano:
            logger.warning("No se pudieron extraer todos los datos requeridos")
            return {
                "success": False,
                "error": "No se pudieron extraer los datos completos de la INE",
                "details": {
                    "cic_found": bool(cic),
                    "id_ciudadano_found": bool(id_ciudadano),
                    "front_text": front_data.get("text_extracted", ""),
                    "back_text": back_data.get("text_extracted", "")
                }
            }
        
        logger.info(f"Datos extraídos exitosamente - CIC: {cic}, ID: {id_ciudadano}")
        
        return {
            "success": True,
            "data": {
                "cic": cic,
                "id_ciudadano": id_ciudadano
            },
            "source": {
                "cic_from": "front" if front_data.get("cic") else "back",
                "id_from": "front" if front_data.get("id_ciudadano") else "back"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error inesperado: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)