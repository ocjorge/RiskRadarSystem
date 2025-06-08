import cv2
import torch
import numpy as np
import time
import json
import csv
import os
from datetime import datetime
from ultralytics import YOLO
from collections import Counter
import matplotlib.pyplot as plt


# --- Sugerencia: Alertas Auditivas (Opcional) ---
# Si quieres alertas de voz, instala pyttsx3: pip install pyttsx3
# Luego, descomenta las siguientes líneas y las llamadas a self._speak()
try:
     import pyttsx3
     TTS_ENABLED = True
except ImportError:
     print("Advertencia: Librería pyttsx3 no encontrada. Las alertas de voz están deshabilitadas.")
     TTS_ENABLED = False

class RiskRadarSystem:
    def __init__(self, model_path_vehicles, video_path, output_dir, config):
        """
        Inicializar el sistema de radar de riesgo para ciclistas.

        Args:
            model_path_vehicles (str): Ruta al modelo YOLO entrenado para vehículos.
            video_path (str): Ruta al video de entrada.
            output_dir (str): Directorio de salida para resultados.
            config (dict): Diccionario con todos los parámetros de configuración.
        """
        self.video_path = video_path
        self.output_dir = output_dir
        self.config = config

        # Asignar configuración a atributos de la clase para fácil acceso
        self.model_path_vehicles = model_path_vehicles
        self.model_path_coco = 'yolov8n.pt'  # Modelo general para otros objetos
        self.CONFIDENCE_THRESHOLD = config['YOLO_CONFIDENCE_THRESHOLD']

        # Crear directorio de salida
        os.makedirs(output_dir, exist_ok=True)

        # Datos de análisis
        self.detection_data = []
        self.frame_stats = []
        self.processing_times = []
        self.risk_history = []

        # Cargar modelos, configurar video, logging y componentes de riesgo
        self._load_models()
        self._setup_video()
        self._setup_risk_components()
        self._setup_logging()

        # --- Configuración del motor de voz (Opcional) ---
        # if TTS_ENABLED:
        #     self.tts_engine = pyttsx3.init()
        #     self.last_alert_level = 'Bajo'

    def _load_models(self):
        """Cargar modelos YOLO y MiDaS."""
        print("Cargando modelos...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Usando dispositivo: {self.device}")

        # Cargar YOLO para vehículos (personalizado)
        self.model_vehicles = YOLO(self.model_path_vehicles)
        print(f"Modelo de vehículos cargado: {self.model_path_vehicles}")

        # Cargar YOLO para objetos generales (COCO)
        try:
            self.model_coco = YOLO(self.model_path_coco)
            print(f"Modelo COCO cargado: {self.model_path_coco}")
        except Exception as e:
            print(
                f"Advertencia: No se pudo cargar el modelo COCO '{self.model_path_coco}'. Se continuará sin él. Error: {e}")
            self.model_coco = None

        # Cargar MiDaS para estimación de profundidad
        self.midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
        self.midas.to(self.device)
        self.midas.eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = midas_transforms.dpt_transform
        print("Modelos cargados exitosamente.")

    def _setup_video(self):
        """Configurar captura y escritura de video."""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el video: {self.video_path}")

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out_video = cv2.VideoWriter(
            os.path.join(self.output_dir, 'output_risk_radar.mp4'),
            fourcc, self.fps, (self.frame_width, self.frame_height)
        )

    def _setup_risk_components(self):
        """Inicializar el mapa de calor y la máscara del cono."""
        # Mapa de calor de baja resolución para rendimiento
        heatmap_h = int(self.frame_height * self.config['HEATMAP_RESOLUTION_FACTOR'])
        heatmap_w = int(self.frame_width * self.config['HEATMAP_RESOLUTION_FACTOR'])
        self.risk_heatmap = np.zeros((heatmap_h, heatmap_w), dtype=np.float32)

        # Crear máscara del cono de riesgo (una sola vez)
        self.cone_mask = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
        bottom_y = int(self.frame_height * self.config['CONE_BOTTOM_Y_FACTOR'])
        top_y = 0
        top_width = self.frame_width * self.config['CONE_TOP_WIDTH_FACTOR']

        p1 = (int(self.frame_width / 2 - top_width / 2), top_y)
        p2 = (int(self.frame_width / 2 + top_width / 2), top_y)
        p3 = (self.frame_width, bottom_y)
        p4 = (0, bottom_y)

        # Corrección para un cono más realista desde el centro inferior
        p3_cone = (int(self.frame_width / 2 + 20), bottom_y)  # Un poco de ancho en la base
        p4_cone = (int(self.frame_width / 2 - 20), bottom_y)

        # Definimos los puntos del polígono para el cono
        cone_points = np.array([p1, p2, p3_cone, p4_cone], np.int32)

        cv2.fillPoly(self.cone_mask, [cone_points], 255)
        self.cone_mask_low_res = cv2.resize(self.cone_mask, (heatmap_w, heatmap_h), interpolation=cv2.INTER_NEAREST) > 0

    def _setup_logging(self):
        """Configurar archivo de log."""
        self.log_file = os.path.join(self.output_dir, 'processing_log.txt')
        with open(self.log_file, 'w') as f:
            f.write(f"Inicio del procesamiento: {datetime.now()}\n")
            f.write(json.dumps(self.config, indent=4) + "\n")
            f.write("-" * 50 + "\n")

    def _log_message(self, message):
        """Escribir mensaje en log y consola."""
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n")

    def _add_heat(self, center_x, center_y, radius, value):
        """Añade calor a una región circular del heatmap."""
        h, w = self.risk_heatmap.shape
        # Crear coordenadas para toda la matriz
        y, x = np.ogrid[:h, :w]
        # Crear una máscara circular
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        mask = dist_from_center <= radius
        # Añadir el valor solo dentro de la máscara
        self.risk_heatmap[mask] += value

    def _process_frame(self, frame, frame_number, timestamp):
        """Procesa un frame, actualiza el mapa de calor y genera la visualización."""
        frame_start_time = time.time()

        # 1. Enfriar el mapa de calor
        self.risk_heatmap *= self.config['HEATMAP_DECAY_RATE']

        # 2. Obtener mapa de profundidad con MiDaS
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(img_rgb).to(self.device)
        if input_tensor.dim() == 3: input_tensor = input_tensor.unsqueeze(0)

        with torch.no_grad():
            prediction = self.midas(input_tensor)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1), size=img_rgb.shape[:2], mode="bicubic", align_corners=False,
            ).squeeze()
        depth_map = prediction.cpu().numpy()

        # 3. Detección de objetos con ambos modelos
        all_detections = []
        # Modelo de vehículos
        results_v = self.model_vehicles.predict(source=frame, conf=self.CONFIDENCE_THRESHOLD, verbose=False)
        if results_v[0].boxes:
            for box in results_v[0].boxes:
                all_detections.append({'box': box, 'names': self.model_vehicles.names})
        # Modelo COCO
        if self.model_coco:
            results_c = self.model_coco.predict(source=frame, conf=self.CONFIDENCE_THRESHOLD,
                                                classes=self.config['COCO_CLASSES_TO_SEEK_IDS'], verbose=False)
            if results_c[0].boxes:
                for box in results_c[0].boxes:
                    all_detections.append({'box': box, 'names': self.model_coco.names})

        # 4. Actualizar mapa de calor y recolectar datos
        frame_detections_data = []
        for det in all_detections:
            box_data = det['box']
            x1, y1, x2, y2 = map(int, box_data.xyxy[0])
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            # ### CORRECCIÓN 2: Definir class_name y otros datos ANTES del if ###
            class_id = int(box_data.cls[0])
            class_name = det['names'][class_id]
            confidence = float(box_data.conf[0])

            # Guardar datos de todas las detecciones (para reportes)
            detection_info = {
                'frame_number': frame_number, 'timestamp': timestamp, 'class': class_name,
                'confidence': confidence, 'bbox': [x1, y1, x2, y2]
            }
            frame_detections_data.append(detection_info)

            # Solo procesar objetos dentro del cono de riesgo para el heatmap
            if 0 <= cy < self.frame_height and 0 <= cx < self.frame_width and self.cone_mask[cy, cx] == 255:
                # Estimar distancia desde el mapa de profundidad
                depth_roi = depth_map[y1:y2, x1:x2]
                median_depth_value = np.median(depth_roi) if depth_roi.size > 0 else 0

                # Añadir "calor" al mapa
                base_heat = self.config['HEAT_INTENSITY_FACTORS'].get(class_name, 0.3)
                depth_factor = np.clip(median_depth_value / 50.0, 0.1, 2.0)
                heat_to_add = base_heat * depth_factor

                # Añadir el calor en la posición correspondiente del mapa de baja resolución
                hm_cx = int(cx * self.config['HEATMAP_RESOLUTION_FACTOR'])
                hm_cy = int(cy * self.config['HEATMAP_RESOLUTION_FACTOR'])

                # ### CORRECCIÓN 1: Usar la nueva función para añadir calor ###
                self._add_heat(center_x=hm_cx, center_y=hm_cy, radius=5, value=heat_to_add)

        self.detection_data.extend(frame_detections_data)

        # 5. Evaluar el riesgo global y generar alertas
        total_heat_in_cone = np.sum(self.risk_heatmap[self.cone_mask_low_res])
        risk_level = "Bajo"
        risk_color = (0, 255, 0)
        if total_heat_in_cone > self.config['HEAT_THRESHOLD_HIGH']:
            risk_level = "Alto"
            risk_color = (0, 0, 255)
        elif total_heat_in_cone > self.config['HEAT_THRESHOLD_MEDIUM']:
            risk_level = "Medio"
            risk_color = (0, 165, 255)

        self.risk_history.append(risk_level)

        # 6. Crear el frame final anotado
        annotated_frame = self._visualize_frame(frame, all_detections, risk_level, risk_color, total_heat_in_cone)

        # 7. Guardar estadísticas del frame
        processing_time = time.time() - frame_start_time
        self.processing_times.append(processing_time)
        frame_stats = {
            'frame_number': frame_number, 'timestamp': timestamp, 'detection_count': len(all_detections),
            'processing_time': processing_time, 'total_heat': total_heat_in_cone, 'risk_level': risk_level
        }
        self.frame_stats.append(frame_stats)

        return annotated_frame

    def _visualize_frame(self, frame, detections, risk_level, risk_color, total_heat):
        """Crea la visualización final combinando todos los elementos."""
        vis_frame = frame.copy()

        # Dibujar heatmap
        heatmap_upscaled = cv2.resize(self.risk_heatmap, (self.frame_width, self.frame_height))
        heatmap_normalized = cv2.normalize(heatmap_upscaled, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)

        # Aplicar máscara del cono al heatmap y mezclar con el frame
        masked_heatmap = cv2.bitwise_and(heatmap_colored, heatmap_colored, mask=self.cone_mask)
        vis_frame = cv2.addWeighted(vis_frame, 0.7, masked_heatmap, 0.5, 0)

        # Dibujar contorno del cono
        cv2.polylines(vis_frame,
                      [np.array(cv2.findContours(self.cone_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0][0])],
                      isClosed=True, color=(255, 255, 0), thickness=2)

        # Dibujar Bounding Boxes de las detecciones
        for det in detections:
            box_data = det['box']
            x1, y1, x2, y2 = map(int, box_data.xyxy[0])
            class_name = det['names'][int(box_data.cls[0])]
            conf = float(box_data.conf[0])
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (200, 200, 0), 1)
            cv2.putText(vis_frame, f"{class_name} {conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255, 255), 1)

        # Dibujar banner de alerta
        cv2.rectangle(vis_frame, (0, 0), (self.frame_width, 40), (0, 0, 0), -1)
        cv2.putText(vis_frame, f"NIVEL DE RIESGO: {risk_level.upper()}", (10, 28), cv2.FONT_HERSHEY_DUPLEX, 1,
                    risk_color, 2)
        cv2.putText(vis_frame, f"Heat: {total_heat:.2f}", (self.frame_width - 200, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 1)

        return vis_frame

    # def _speak(self, text):
    #     """Función para vocalizar alertas."""
    #     if TTS_ENABLED:
    #         self.tts_engine.say(text)
    #         self.tts_engine.runAndWait()

    def process_video(self):
        """Procesar el video completo, frame por frame."""
        self._log_message("Iniciando procesamiento de video...")
        start_time = time.time()
        frame_count = 0

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            timestamp = frame_count / self.fps

            try:
                processed_frame = self._process_frame(frame, frame_count, timestamp)
                self.out_video.write(processed_frame)
            except Exception as e:
                self._log_message(f"Error procesando frame {frame_count}: {e}")
                import traceback
                traceback.print_exc()
                continue

            frame_count += 1
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                avg_fps = frame_count / elapsed
                self._log_message(f"Procesados {frame_count}/{self.total_frames} frames | Velocidad: {avg_fps:.2f} FPS")

        # Finalizar
        self._log_message(f"Procesamiento completado en {time.time() - start_time:.1f}s")
        self._cleanup()
        self._generate_reports()

    def _cleanup(self):
        """Liberar recursos."""
        self.cap.release()
        self.out_video.release()
        cv2.destroyAllWindows()

    def _generate_reports(self):
        """Generar todos los reportes y visualizaciones finales."""
        self._log_message("Generando reportes finales...")

        # Guardar datos en JSON y CSV (código heredado y útil)
        with open(os.path.join(self.output_dir, 'detections_raw.json'), 'w') as f:
            json.dump(self.detection_data, f, indent=2)

        if self.detection_data:
            with open(os.path.join(self.output_dir, 'detections_raw.csv'), 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.detection_data[0].keys())
                writer.writeheader()
                writer.writerows(self.detection_data)

        # Generar estadísticas y reporte de texto
        self._generate_statistics_report()

        # Generar visualizaciones
        self._generate_visualizations()
        self._log_message("Reportes generados exitosamente.")

    def _generate_statistics_report(self):
        """Generar estadísticas incluyendo el análisis de riesgo."""
        stats = {
            'video_info': {
                'path': self.video_path,
                'resolution': f"{self.frame_width}x{self.frame_height}",
                'fps': self.fps,
                'total_frames': self.total_frames
            },
            'processing_info': {
                'frames_processed': len(self.frame_stats),
                'avg_processing_time_per_frame': np.mean(self.processing_times) if self.processing_times else 0,
                'total_processing_time': sum(self.processing_times)
            },
            'detection_stats': {},
            'risk_analysis': {}
        }

        if self.detection_data:
            stats['detection_stats']['by_class'] = dict(Counter([d['class'] for d in self.detection_data]))

        if self.risk_history:
            risk_counts = Counter(self.risk_history)
            total_risk_frames = len(self.risk_history)
            stats['risk_analysis']['time_in_risk_level_percent'] = {
                level: (count / total_risk_frames) * 100 for level, count in risk_counts.items()
            }
            stats['risk_analysis']['risk_level_counts'] = dict(risk_counts)

        with open(os.path.join(self.output_dir, 'statistics_report.json'), 'w') as f:
            json.dump(stats, f, indent=4)

        # Reporte de texto
        with open(os.path.join(self.output_dir, 'summary_report.txt'), 'w') as f:
            f.write("REPORTE DE ANÁLISIS DE RIESGO\n")
            f.write("=" * 50 + "\n\n")
            f.write("ANÁLISIS DE RIESGO:\n")
            for level, perc in stats['risk_analysis'].get('time_in_risk_level_percent', {}).items():
                f.write(f"  - Tiempo en Riesgo '{level}': {perc:.2f}%\n")
            f.write("\nDETECCIONES POR CLASE:\n")
            for class_name, count in stats['detection_stats'].get('by_class', {}).items():
                f.write(f"  - {class_name}: {count}\n")
            f.write(f"\nPROCESAMIENTO:\n  - Frames procesados: {stats['processing_info']['frames_processed']}\n")
            f.write(
                f"  - Tiempo promedio por frame: {stats['processing_info']['avg_processing_time_per_frame']:.3f}s\n")

    def _generate_visualizations(self):
        """Generar gráficos de análisis, incluyendo el riesgo."""
        if not self.frame_stats: return

        plt.style.use('seaborn-v0_8-darkgrid')
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('Análisis del Procesamiento de Riesgo', fontsize=16)

        # 1. Distribución de clases detectadas
        if self.detection_data:
            class_counts = Counter([d['class'] for d in self.detection_data])
            axes[0, 0].bar(class_counts.keys(), class_counts.values(), color='skyblue')
            axes[0, 0].set_title('Distribución de Detecciones por Clase')
            axes[0, 0].tick_params(axis='x', rotation=45)

        # 2. Línea de tiempo del "Calor" Total en el Cono
        total_heat_history = [s['total_heat'] for s in self.frame_stats]
        axes[0, 1].plot(total_heat_history, color='orangered')
        axes[0, 1].set_title('Nivel de "Calor" en el Cono a lo Largo del Tiempo')
        axes[0, 1].set_xlabel('Número de Frame')
        axes[0, 1].set_ylabel('Calor Total (Unidad Arbitraria)')

        # 3. Distribución del Tiempo en Niveles de Riesgo
        if self.risk_history:
            risk_counts = Counter(self.risk_history)
            axes[1, 0].pie(risk_counts.values(), labels=risk_counts.keys(), autopct='%1.1f%%',
                           colors=[{'Bajo': 'green', 'Medio': 'orange', 'Alto': 'red'}[key] for key in
                                   risk_counts.keys()])
            axes[1, 0].set_title('Distribución de Tiempo por Nivel de Riesgo')

        # 4. Tiempo de procesamiento por frame
        axes[1, 1].plot(self.processing_times, color='purple', alpha=0.7)
        axes[1, 1].set_title('Tiempo de Procesamiento por Frame')
        axes[1, 1].set_xlabel('Número de Frame')
        axes[1, 1].set_ylabel('Segundos')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(self.output_dir, 'analysis_charts.png'), dpi=300)
        plt.close()


def main():
    """Función principal para configurar y ejecutar el sistema."""

    # ==============================================================================
    # CONFIGURACIÓN PRINCIPAL - ¡AJUSTA ESTOS VALORES!
    # ==============================================================================
    config = {
        # --- Rutas ---
        'MODEL_PATH_VEHICLES': 'F:/Documents/PycharmProjects/DepthDetector/best.pt',
        'VIDEO_INPUT_PATH': 'F:/Documents/PycharmProjects/DepthDetector/GH012372_no_audio.mp4',
        'OUTPUT_DIR': 'results_risk_radar',

        # --- Parámetros de Detección ---
        'YOLO_CONFIDENCE_THRESHOLD': 0.40,
        'COCO_CLASSES_TO_SEEK_IDS': [0, 1, 16],  # 0: person, 1: bicycle, 16: dog

        # --- Parámetros del Cono de Riesgo ---
        'CONE_BOTTOM_Y_FACTOR': 0.95,  # Dónde nace el cono (0.0=arriba, 1.0=abajo)
        'CONE_TOP_WIDTH_FACTOR': 0.8,  # Ancho del cono en la parte superior (0.0 a 1.0)

        # --- Parámetros del Mapa de Calor ---
        'HEATMAP_RESOLUTION_FACTOR': 0.25,  # Procesar heatmap a 1/4 de resolución para rendimiento
        'HEATMAP_DECAY_RATE': 0.92,  # Factor de enfriamiento por frame (más bajo = enfría más rápido)
        'HEAT_INTENSITY_FACTORS': {  # "Peligro" base por clase
            'car': 0.8, 'threewheel': 0.7, 'bus': 1.0, 'truck': 1.0,
            'motorbike': 0.6, 'van': 0.9, 'person': 0.4, 'bicycle': 0.3, 'dog': 0.5
        },

        # --- Parámetros de Alerta ---
        'HEAT_THRESHOLD_MEDIUM': 15.0,  # Umbral de calor total para alerta Media
        'HEAT_THRESHOLD_HIGH': 30.0,  # Umbral de calor total para alerta Alta
    }

    # Verificar que los archivos existen
    if not os.path.exists(config['MODEL_PATH_VEHICLES']):
        print(f"Error: No se encontró el modelo en {config['MODEL_PATH_VEHICLES']}")
        return
    if not os.path.exists(config['VIDEO_INPUT_PATH']):
        print(f"Error: No se encontró el video en {config['VIDEO_INPUT_PATH']}")
        return

    try:
        # Crear y ejecutar el sistema
        radar = RiskRadarSystem(
            config['MODEL_PATH_VEHICLES'],
            config['VIDEO_INPUT_PATH'],
            config['OUTPUT_DIR'],
            config
        )
        radar.process_video()

        print(f"\n✅ Procesamiento completado exitosamente!")
        print(f"📁 Resultados guardados en: {os.path.abspath(config['OUTPUT_DIR'])}/")

    except Exception as e:
        print(f"❌ Error catastrófico durante el procesamiento: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
