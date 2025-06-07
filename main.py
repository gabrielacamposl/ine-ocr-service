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
import uvicorn
import os
    
# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="INE OCR Microservice - MEJORADO",
    description="Microservicio para extraer CIC e Identificador del Ciudadano de INE mexicana con patrones optimizados",
    version="2.0.0"
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
    """Clase principal para procesar imágenes de INE y extraer datos - VERSIÓN MEJORADA"""
    
    def __init__(self):
        # Configurar pytesseract
        pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'  # Para Railway/Docker
        
        self.patterns = {
            'cic': [
                r'IDMEX(\d{9})',                    # Patrón principal: 9 dígitos después de IDMEX
                r'IDMEX(\d{9})\d*',                 # Alternativo: primeros 9 dígitos después de IDMEX
                r'CIC[:\s]*(\d{9,10})',             # CIC precedido por "CIC"
            ],
            'id_ciudadano': [
                r'IDMEX\d+<<\d*(\d{9})(?:\s|$)',    # Últimos 9 dígitos de la línea IDMEX
                r'IDMEX\d+<<\d*(\d{9})',            # Alternativo sin fin de línea
                r'<<\d*(\d{9})(?:\s|$)',            # Últimos 9 dígitos después de <<
                r'(\d{9})(?:\s|$)',                 # Fallback: cualquier secuencia de 9 dígitos al final
            ],
            'curp': [
                r'([A-Z]{4}\d{6}[HM][A-Z]{5}[A-Z\d]{2})',  # Patrón CURP estándar
                r'CURP[:\s]*([A-Z]{4}\d{6}[HM][A-Z]{5}[A-Z\d]{2})',
                r'^[A-Z]{1}[AEIOU]{1}[A-Z]{2}\d{2}(0[1-9]|1[0-2])(0[1-9]|[12][0-9]|3[01])[HM]{1}(AS|BC|BS|CC|CL|CM|CS|CH|DF|DG|GT|GR|HG|JC|MC|MN|MS|NT|NL|OC|PL|QT|QR|SP|SL|SR|TC|TS|TL|VZ|YN|ZS|NE)[B-DF-HJ-NP-TV-Z]{3}[A-Z\d]{1}[A-Z\d]{1}$',
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
        """Preprocesa la imagen para mejorar el OCR, especialmente números"""
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
            
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            
            # Aplicar filtro bilateral para reducir ruido manteniendo bordes nítidos
            denoised = cv2.bilateralFilter(blurred, 9, 75, 75)
            
            # Mejorar contraste usando CLAHE con parámetros optimizados para texto
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)
            
            # Aplicar unsharp masking para hacer los números más nítidos
            gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
            unsharp_mask = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
            
            # Aplicar threshold adaptativo con parámetros optimizados para números
            binary = cv2.adaptiveThreshold(
                unsharp_mask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 
                blockSize=15,  # Aumentado para mejor detección de números
                C=10           # Ajustado para mejor contraste
            )
            
            kernel_small = np.ones((1,1), np.uint8)
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_small)
            
            # Aplicar opening para separar números que puedan estar conectados
            kernel_open = np.ones((2,2), np.uint8)
            opened = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_open)
            
            # Dilatar ligeramente para hacer números más gruesos y legibles
            kernel_dilate = np.ones((1,1), np.uint8)
            final_image = cv2.dilate(opened, kernel_dilate, iterations=1)
            
            return final_image
            
        except Exception as e:
            logger.error(f"Error en preprocesamiento: {e}")
            return image
    
    def extract_text_regions(self, image: np.ndarray) -> list:
        """Extrae regiones de texto específicas de la INE, optimizado para datos traseros"""
        try:
            height, width = image.shape
            
            # 📍 REGIONES ESPECÍFICAS PARA INE (enfocadas en parte trasera)
            regions = [
                # Región específica para línea IDMEX (parte inferior central)
                {"name": "idmex_region", "roi": (0, int(height*0.7), width, int(height*0.95))},
                
                # Región para códigos inferiores (donde aparecen CIC e ID)
                {"name": "bottom_codes", "roi": (0, int(height*0.6), width, height)},
                
                # Región central amplia (para capturar datos principales)
                {"name": "center_wide", "roi": (0, int(height*0.3), width, int(height*0.8))},
                
                # Región superior (para CURP y otros datos)
                {"name": "top_data", "roi": (0, 0, width, int(height*0.4))},
                
                # Región central específica (datos del medio)
                {"name": "center", "roi": (0, int(height*0.4), width, int(height*0.7))},
                
                # Toda la imagen como fallback final
                {"name": "full", "roi": (0, 0, width, height)}
            ]
            
            extracted_regions = []
            
            for region in regions:
                x, y, x2, y2 = region["roi"]
                
                # Asegurar que las coordenadas estén dentro de los límites
                x = max(0, x)
                y = max(0, y)
                x2 = min(width, x2)
                y2 = min(height, y2)
                
                # Extraer ROI
                roi = image[y:y2, x:x2]
                
                if roi.size > 0:
                    # Verificar que la región tenga contenido útil
                    if roi.shape[0] > 10 and roi.shape[1] > 10:  # Mínimo 10x10 pixels
                        extracted_regions.append({
                            "name": region["name"],
                            "image": roi,
                            "coordinates": (x, y, x2, y2)
                        })
                        logger.debug(f"Región extraída: {region['name']} - {roi.shape}")
            
            return extracted_regions
            
        except Exception as e:
            logger.error(f"Error extrayendo regiones: {e}")
            # Fallback seguro
            return [{"name": "full", "image": image, "coordinates": (0, 0, width, height)}]
    
    def perform_ocr(self, image: np.ndarray) -> str:
        """Realiza OCR en la imagen usando configuraciones optimizadas para números"""
        try:
            # 🔍 CONFIGURACIONES DE OCR OPTIMIZADAS PARA NÚMEROS EN INE
            configs = [
                # Configuración específica para números con caracteres permitidos
                '--psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<>',
                
                # Configuración para líneas de texto con números
                '--psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<>',
                
                # Configuración para palabras individuales (útil para IDMEX)
                '--psm 8 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<>',
                
                # Configuración específica solo para números (para códigos)
                '--psm 6 -c tessedit_char_whitelist=0123456789<>',
                '--psm 7 -c tessedit_char_whitelist=0123456789<>',
                
                # Configuraciones con diferentes PSM para mejor reconocimiento
                '--psm 6 -c tessedit_char_blacklist=!@#$%^&*()_+=[]{}|;:,.<>?`~',
                '--psm 13 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<>',
                
                # Configuración general como fallback
                '--psm 6',
                '--psm 4',
            ]
            
            results = []
            
            for config in configs:
                try:
                    # Usar idioma español para mejor reconocimiento
                    text = pytesseract.image_to_string(image, config=config, lang='spa+eng')
                    if text.strip():
                        # Limpiar el texto extraído
                        cleaned_text = self.clean_ocr_text(text.strip())
                        if cleaned_text:
                            results.append(cleaned_text)
                except Exception as e:
                    logger.debug(f"Error en configuración OCR: {e}")
                    continue
            
            # Combinar todos los resultados
            combined_text = '\n'.join(results)
            return combined_text
            
        except Exception as e:
            logger.error(f"Error en OCR: {e}")
            return ""
    
    def clean_ocr_text(self, text: str) -> str:
        """Limpia el texto OCR y corrige errores comunes de números"""
        try:
            # 🧹 CORRECCIONES ESPECÍFICAS PARA NÚMEROS CONFUNDIDOS
            replacements = {
                # Correcciones específicas para números confundidos
                'l': '1',     # l minúscula -> 1
                'I': '1',     # I mayúscula -> 1
                'O': '0',     # O mayúscula -> 0
                'o': '0',     # o minúscula -> 0
                'S': '5',     # S -> 5 (en algunos casos)
                'Z': '2',     # Z -> 2 (en algunos casos)
                'B': '8',     # B -> 8 (en algunos casos)
                'G': '6',     # G -> 6 (en algunos casos)
                
                # Limpiar caracteres extraños
                ' ': '',      # Eliminar espacios
                '\n': ' ',    # Convertir saltos de línea a espacios
                '\t': ' ',    # Convertir tabs a espacios
            }
            
            cleaned = text
            for old, new in replacements.items():
                cleaned = cleaned.replace(old, new)
            
            # Limpiar espacios múltiples
            cleaned = re.sub(r'\s+', ' ', cleaned)
            
            return cleaned.strip()
            
        except Exception as e:
            logger.error(f"Error limpiando texto OCR: {e}")
            return text
    
    def extract_data_with_patterns(self, text: str) -> Dict[str, Optional[str]]:
        """Extrae CIC, ID Ciudadano y CURP usando patrones regex optimizados"""
        extracted_data = {
            'cic': None,
            'id_ciudadano': None,
            'curp': None
        }
        
        lines = text.strip().split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Limpiar cada línea pero mantener estructura
            cleaned_line = re.sub(r'\s+', '', line.upper().strip())
            if cleaned_line:
                cleaned_lines.append(cleaned_line)
        
        # Combinar en texto limpio para búsqueda general
        cleaned_text = ' '.join(cleaned_lines)
        
        logger.info(f"Texto limpio para análisis: {cleaned_text}")
        
        cic_found = False
        for line in cleaned_lines:
            if 'IDMEX' in line:
                logger.info(f"Línea con IDMEX encontrada: {line}")
                for pattern in self.patterns['cic']:
                    matches = re.findall(pattern, line)
                    if matches:
                        cic_candidate = matches[0]
                        if len(cic_candidate) >= 9:
                            # Tomar solo los primeros 9 dígitos
                            extracted_data['cic'] = cic_candidate[:9]
                            logger.info(f"CIC extraído: {extracted_data['cic']} usando patrón: {pattern}")
                            cic_found = True
                            break
                if cic_found:
                    break

        id_found = False
        for line in cleaned_lines:
            if 'IDMEX' in line:
                logger.info(f"Analizando línea IDMEX para ID Ciudadano: {line}")
                
                # Buscar patrón específico: IDMEX[números]<<[números][los últimos 9 dígitos]
                for pattern in self.patterns['id_ciudadano']:
                    matches = re.findall(pattern, line)
                    if matches:
                        id_candidate = matches[0]
                        if len(id_candidate) == 9:
                            extracted_data['id_ciudadano'] = id_candidate
                            logger.info(f"ID Ciudadano extraído: {extracted_data['id_ciudadano']} usando patrón: {pattern}")
                            id_found = True
                            break

                if not id_found:
                    # Buscar << y tomar los últimos 9 dígitos de la línea
                    if '<<' in line:
                        # Dividir por << y tomar la parte después
                        parts = line.split('<<')
                        if len(parts) > 1:
                            after_brackets = parts[-1]  # Última parte después de <<
                            # Extraer todos los números de esta parte
                            numbers = re.findall(r'\d+', after_brackets)
                            if numbers:
                                # Tomar el último grupo de números y obtener los últimos 9 dígitos
                                last_numbers = numbers[-1]
                                if len(last_numbers) >= 9:
                                    extracted_data['id_ciudadano'] = last_numbers[-9:]
                                    logger.info(f"ID Ciudadano extraído (fallback): {extracted_data['id_ciudadano']}")
                                    id_found = True
                
                if id_found:
                    break
        
        # 🎯 EXTRAER CURP
        for line in cleaned_lines:
            for pattern in self.patterns['curp']:
                matches = re.findall(pattern, line)
                if matches:
                    curp_candidate = matches[0]
                    if len(curp_candidate) == 18:  # CURP debe tener exactamente 18 caracteres
                        extracted_data['curp'] = curp_candidate
                        logger.info(f"CURP extraído: {extracted_data['curp']}")
                        break
            if extracted_data['curp']:
                break
        
        if extracted_data['cic'] and len(extracted_data['cic']) != 9:
            logger.warning(f"CIC tiene longitud incorrecta: {len(extracted_data['cic'])}")
            extracted_data['cic'] = None
            
        if extracted_data['id_ciudadano'] and len(extracted_data['id_ciudadano']) != 9:
            logger.warning(f"ID Ciudadano tiene longitud incorrecta: {len(extracted_data['id_ciudadano'])}")
            extracted_data['id_ciudadano'] = None
        
        return extracted_data
    
    def process_ine_image(self, image_file) -> Dict[str, Optional[str]]:
        """Procesa una imagen de INE y extrae los datos principales"""
        try:
            logger.info("🔍 Iniciando procesamiento de imagen INE con algoritmos mejorados")
            
            # Leer imagen desde archivo
            image_data = image_file.read()
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("No se pudo decodificar la imagen")
            
            # Preprocesar imagen con mejoras
            processed_image = self.preprocess_image(image)
            
            # Extraer regiones de texto optimizadas
            regions = self.extract_text_regions(processed_image)
            
            all_extracted_data = {
                'cic': None,
                'id_ciudadano': None,
                'curp': None
            }
            
            # Procesar cada región con prioridad
            for region in regions:
                logger.info(f"🔍 Procesando región: {region['name']}")
                
                # Realizar OCR mejorado
                text = self.perform_ocr(region['image'])
                
                if text:
                    logger.info(f"📝 Texto extraído de {region['name']}: {text[:100]}...")
                    
                    # Extraer datos con patrones mejorados
                    extracted_data = self.extract_data_with_patterns(text)
                    
                    # Actualizar datos si encontramos algo nuevo
                    for key, value in extracted_data.items():
                        if value and not all_extracted_data[key]:
                            all_extracted_data[key] = value
                            logger.info(f"✅ Encontrado {key}: {value}")
                
                # Si ya tenemos CIC e ID, podemos parar (optimización)
                if all_extracted_data['cic'] and all_extracted_data['id_ciudadano']:
                    logger.info("🎯 CIC e ID Ciudadano encontrados, optimizando búsqueda")
                    break
            
            # Limpieza de memoria
            self.aggressive_memory_cleanup()
            
            logger.info(f"🏁 Procesamiento completado: CIC={all_extracted_data['cic']}, ID={all_extracted_data['id_ciudadano']}, CURP={all_extracted_data['curp']}")
            
            return all_extracted_data
            
        except Exception as e:
            logger.error(f"❌ Error procesando imagen INE: {e}")
            raise HTTPException(status_code=500, detail=f"Error procesando imagen: {str(e)}")

# Instancia del procesador mejorado
ine_processor = INEProcessor()

@app.get("/")
async def root():
    return {
        "message": "INE OCR Microservice - VERSIÓN MEJORADA", 
        "status": "active",
        "version": "2.0.0",
        "improvements": "Patrones CIC/ID optimizados + mejor reconocimiento números"
    }

@app.get("/health")
async def health_check():
    try:
        # Verificar que tesseract funcione
        tesseract_version = pytesseract.get_tesseract_version()
        
        return {
            "status": "healthy", 
            "service": "ine-ocr",
            "version": "2.0.0",
            "tesseract_version": str(tesseract_version),
            "improvements": [
                "Patrones CIC/ID optimizados para formato real INE",
                "Reconocimiento mejorado de números 1/7",
                "Preprocesamiento avanzado de imagen",
                "Configuraciones OCR específicas para números"
            ]
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "ine-ocr", 
            "error": str(e)
        }

@app.post("/extract-ine-data")
async def extract_ine_data(
    ine_front: UploadFile = File(..., description="Imagen frontal de la INE"),
    ine_back: UploadFile = File(None, description="Imagen trasera de la INE (opcional)")
):
   
    try:
        # Validar archivo frontal
        if not ine_front.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="El archivo frontal debe ser una imagen")
        
        # Procesar imagen frontal
        logger.info("🔍 Procesando imagen frontal de INE con algoritmos mejorados")
        front_data = ine_processor.process_ine_image(ine_front.file)
        
        result = {
            "success": True,
            "data": {
                "cic": front_data.get('cic'),
                "id_ciudadano": front_data.get('id_ciudadano'),
                "curp": front_data.get('curp')
            },
            "processed_images": ["front"],
            "version": "2.0.0"
        }
        
        # Si se proporcionó imagen trasera, procesarla también
        if ine_back and ine_back.content_type.startswith('image/'):
            logger.info("🔍 Procesando imagen trasera de INE con algoritmos mejorados")
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
                "suggestion": "Verifica que la imagen sea clara, esté bien iluminada y contenga la línea IDMEX",
                "expected_format": "IDMEX2559201236<<0835123456789 (CIC: primeros 9 después de IDMEX, ID: últimos 9)",
                "version": "2.0.0"
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
    VERSIÓN MEJORADA con patrones optimizados
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
            "processed_images": ["front"],
            "version": "2.0.0"
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

    port = int(os.environ.get("PORT", 8000))

    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        log_level="info",
        access_log=True
    )