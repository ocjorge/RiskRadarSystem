# 🚴‍♂️ Risk Radar System for Cyclists

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red.svg)
![YOLO](https://img.shields.io/badge/YOLO-v8-yellow.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Status](https://img.shields.io/badge/status-Active-brightgreen.svg)
![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)

Un sistema avanzado de análisis de riesgo en tiempo real para ciclistas que utiliza inteligencia artificial para detectar vehículos y evaluar niveles de peligro en el entorno ciclista.

## 🎯 Características Principales

- **Detección Multi-Modelo**: Utiliza YOLO personalizado para vehículos y modelo COCO para objetos generales
- **Estimación de Profundidad**: Integración con MiDaS para análisis de distancia
- **Sistema de Alertas Inteligente**: Niveles de riesgo (Bajo, Medio, Alto) basados en análisis de calor
- **Visualización Avanzada**: Mapa de calor superpuesto con cono de riesgo
- **Reportes Detallados**: Estadísticas completas y visualizaciones de análisis
- **Alertas de Voz** (Opcional): Notificaciones auditivas con pyttsx3

## 🛠️ Tecnologías Utilizadas

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=python&logoColor=white)

## 📋 Requisitos del Sistema

### Dependencias Principales
```bash
pip install torch torchvision
pip install ultralytics
pip install opencv-python
pip install numpy
pip install matplotlib
pip install pyttsx3  # Opcional para alertas de voz
```

### Requisitos de Hardware
- **GPU**: NVIDIA CUDA compatible (recomendado)
- **RAM**: Mínimo 8GB, recomendado 16GB
- **Espacio**: 5GB libres para modelos y procesamiento

## 🚀 Instalación

1. **Clonar el repositorio**
```bash
git clone https://github.com/tu-usuario/risk-radar-system.git
cd risk-radar-system
```

2. **Crear entorno virtual**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate  # Windows
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

4. **Descargar modelos**
- Coloca tu modelo YOLO personalizado en la ruta especificada
- El modelo COCO se descarga automáticamente

## ⚙️ Configuración

Edita la configuración principal en el archivo:

```python
config = {
    # Rutas de archivos
    'MODEL_PATH_VEHICLES': 'path/to/your/best.pt',
    'VIDEO_INPUT_PATH': 'path/to/your/video.mp4',
    'OUTPUT_DIR': 'results_risk_radar',
    
    # Parámetros de detección
    'YOLO_CONFIDENCE_THRESHOLD': 0.40,
    'COCO_CLASSES_TO_SEEK_IDS': [0, 1, 16],  # person, bicycle, dog
    
    # Configuración del cono de riesgo
    'CONE_BOTTOM_Y_FACTOR': 0.95,
    'CONE_TOP_WIDTH_FACTOR': 0.8,
    
    # Parámetros del mapa de calor
    'HEATMAP_DECAY_RATE': 0.92,
    'HEAT_THRESHOLD_MEDIUM': 15.0,
    'HEAT_THRESHOLD_HIGH': 30.0,
}
```

## 🎮 Uso

### Ejecución Básica
```bash
python risk_radar_system.py
```

### Estructura de Archivos de Salida
```
results_risk_radar/
├── output_risk_radar.mp4      # Video procesado con visualizaciones
├── detections_raw.json        # Datos de detecciones en JSON
├── detections_raw.csv         # Datos de detecciones en CSV
├── statistics_report.json     # Estadísticas detalladas
├── summary_report.txt         # Resumen en texto plano
├── analysis_charts.png        # Gráficos de análisis
└── processing_log.txt         # Log del procesamiento
```

## 📊 Funcionalidades del Sistema

### 🎯 Sistema de Detección
- **Vehículos**: car, bus, truck, motorbike, van, threewheel
- **Objetos Generales**: person, bicycle, dog
- **Confianza Mínima**: Configurable (por defecto 40%)

### 🌡️ Mapa de Calor Inteligente
- **Enfriamiento Gradual**: Factor de decaimiento configurable
- **Intensidad por Clase**: Diferentes niveles de "peligro" por tipo de objeto
- **Zona de Riesgo**: Cono de análisis centrado en la perspectiva del ciclista

### 📈 Niveles de Alerta
| Nivel | Color | Descripción |
|-------|-------|-------------|
| 🟢 Bajo | Verde | Situación normal, pocos objetos detectados |
| 🟡 Medio | Naranja | Presencia moderada de vehículos |
| 🔴 Alto | Rojo | Alta concentración de vehículos peligrosos |

## 📸 Capturas de Pantalla

### Vista del Sistema en Funcionamiento
El sistema muestra:
- Cono de riesgo superpuesto
- Mapa de calor con intensidad por colores
- Bounding boxes de detecciones
- Banner de nivel de riesgo en tiempo real

## 🔧 Personalización Avanzada

### Ajustar Sensibilidad
```python
'HEAT_INTENSITY_FACTORS': {
    'car': 0.8,        # Vehículos estándar
    'bus': 1.0,        # Vehículos grandes
    'motorbike': 0.6,  # Vehículos pequeños
    'person': 0.4,     # Peatones
}
```

### Modificar Zona de Riesgo
```python
'CONE_BOTTOM_Y_FACTOR': 0.95,  # Posición vertical del cono
'CONE_TOP_WIDTH_FACTOR': 0.8,   # Amplitud del cono
```

## 🤝 Contribución

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📝 Roadmap

- [ ] Integración con cámaras en tiempo real
- [ ] Alertas push a dispositivos móviles
- [ ] Análisis de patrones de tráfico
- [ ] Integración con GPS para alertas geolocalizadas
- [ ] Mejoras en la estimación de velocidad de vehículos

## 🐛 Problemas Conocidos

- El procesamiento puede ser lento en hardware sin GPU
- Requiere modelos YOLO preentrenados para máxima efectividad
- La estimación de profundidad puede variar según condiciones de iluminación

## 📄 Licencia

Distribuido bajo la Licencia MIT. Ve `LICENSE` para más información.

## 👥 Autores

- **Tu Nombre** - *Desarrollo Principal* - [@ocjorge](https://github.com/ocjorge)

## 🙏 Agradecimientos

- [Ultralytics](https://github.com/ultralytics/ultralytics) por YOLO
- [Intel ISL](https://github.com/intel-isl/MiDaS) por MiDaS
- OpenCV community
- PyTorch team

---

<div align="center">

**¿Te gusta este proyecto? ¡Dale una ⭐ si te ha sido útil!**



</div>
