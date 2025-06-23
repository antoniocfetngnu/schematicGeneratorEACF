# 🧠 Software De Generación De Esquemáticos Eléctricos A Partir De Imágenes De Circuitos Montados En Protoboard

![Status](https://img.shields.io/badge/status-completado-brightgreen)
![Hecho con](https://img.shields.io/badge/Hecho%20con-Kotlin%20%7C%20Python%20%7C%20YOLOv8-ff69b4)
![Plataforma](https://img.shields.io/badge/plataforma-Android%20%7C%20Ubuntu-brightgreen)

---

## 📑 Contenidos

- [👤 Autor](#-autor)
- [🎯 Propósito y Beneficios](#-propósito-y-beneficios)
- [⚙️ Cómo Funciona](#-cómo-funciona)
- [🧠 Modelos de Procesamiento](#-modelos-de-procesamiento)
- [🛠️ Tecnologías](#-tecnologías)
- [🚀 Cómo Instalar](#-cómo-instalar)
- [📄 Artículo Científico](#-referencia-al-artículo-científico)
- [🎥 Video Tutorial](#-también-puedes-ver-el-video-tutorialdemostración-de-uso)
- [🤝 Agradecimientos](#-agradecimientos)
- [📫 Contacto](#-contacto)

---

## 👤 Autor

- **Nombre**: *Calderón Flores Enrique Antonio*
- **Carrera**: Ingeniería en Ciencias de la Computación  
- **Materia**: SIS330 - DESARROLLO DE APLICACIONES INTELIGENTES 	  
- **Universidad**: Universidad Mayor, Real y Pontificia de San Francisco Xavier de Chuquisaca

---

## 🎯 Propósito y Beneficios

Este proyecto ofrece un sistema inteligente para generar esquemas eléctricos de circuitos armados completamente en una sola protoboard, utilizando visión por computadora y aprendizaje profundo. Está dirigido a estudiantes y entusiastas de la electrónica.

### ✨ Beneficios

- ✅ **Generación de Esquemas**: Facilita la verificación manual.
- ✅ **Ahorro de Tiempo y Costos**: Reduce errores de montaje.
- ✅ **Apoyo Educativo**: Brinda retroalimentación visual a principiantes.

---

## ⚙️ Cómo Funciona

<p align="center">
  <img src="./resources/ComponentesSoftware.jpg" width="600" alt="Diagrama de Componentes">
</p>
<p align="center"><i>Diagrama de Componentes del Software</i></p>

El sistema combina una aplicación móvil (cliente) y un servidor en una PC para procesar imágenes de protoboards y generar esquemas eléctricos. La app captura imágenes, permite seleccionar tipos de circuitos integrados (DIPs) y muestra el esquema resultante. El servidor ejecuta modelos de IA para procesar la imagen y devolver los resultados.

---

## 🧠 Modelos de Procesamiento

El sistema usa una arquitectura modular con YOLOv8, dividida en 8 etapas, la última etapa son algoritmos (sin IA) que utilizan como entrada las salidas de los anteriores modelos:

1. 🔍 **Detección de Componentes**
2. 🧷 **Segmentación de Patillas Finas**
3. 🔌 **Segmentación de Cables**
4. 🎯 **Segmentacion de Extremos de Cables**
5. 🧠 **Deteccion de Pines (DIPs/Sensores)**
6. 📍 **Detección de Zonas Clave del Protoboard**
7. 🧭 **Detección de Carriles Verticales del protoboard**
8. 🛠️ **Correcciones Espaciales y Geométricas**

---

## 🛠️ Tecnologías

### 📱 Aplicación Móvil

- **Lenguaje**: Kotlin 1.9.24
- **Frameworks**: Android Studio, OpenCV 4.10, Retrofit
- **Funciones**: Captura de imágenes, preprocesamiento, comunicación con el servidor, renderizado visual con Canvas

### 🖥️ Servidor

- **Lenguaje**: Python 3.8+
- **Frameworks**: Flask, YOLOv8 (Ultralytics), OpenCV
- **Sistema operativo**: Ubuntu (recomendado)
- **Salidas**: JSON con datos del circuito + imagen anotada (3000x3000)

---

## 🚀 Cómo Instalar

### 📦 Requisitos

- **Móvil**:
  - Android 8.0+
  - Android Studio
  - Dependencias: OpenCV 4.10, Retrofit

- **Servidor**:
  - Ubuntu/Linux con Python 3.8+
  - Flask, Ultralytics YOLOv8

### 🧪 Instrucciones

> 📂 El servidor se encuentra en `circuitsDetectionServerEACF`  
> 📂 La app móvil está en `schemaitics`

## 📦 Descarga de Modelos

Los modelos se encuentran disponibles aquí:  
🔗 [Descargar desde Google Drive](https://drive.google.com/drive/folders/1_lzV4dt3Pup1IUGmzTJVtqW_tBoS7uIz?usp=sharing)

#### 1. Configurar el Servidor

```bash
cd circuitsDetectionServerEACF

# Crear carpeta para modelos si no existe
mkdir -p modelos

# Instalar dependencias
pip install -r requirements.txt

# Iniciar el servidor
python app.py
```

> 📁 **Nota**: coloca todos los modelos descargados (YOLOv8 `.pt`, etc.) en la carpeta `modelos/`.
> Asegúrate de haberlos obtenido desde el enlace proporcionado en la sección anterior (por ejemplo, Google Drive o Hugging Face).

> 🌐 Verifica que el servidor esté disponible en la red local: `http://<IP_LOCAL>:5000`

#### 2. Configurar la App Móvil

* Abre el proyecto `schemaitics` en Android Studio.
* Edita `Constants.kt` y `network_security_config.xml` para definir la IP del servidor.
* Integra OpenCV: importa el módulo oficial y sincroniza las versiones en `build.gradle`.
* Compila e instala la app en un dispositivo Android.

#### 3. Uso

1. Abre la app y captura una imagen del circuito.
2. El servidor procesa la imagen.
3. Si hay DIPs, selecciona el tipo (ej. 7408) en el menú desplegable.
4. El servidor envía un JSON con el esquema.
5. La app renderiza el esquema sobre la imagen original usando Canvas.

> 📶 Asegúrate de que el celular y el servidor estén en la **misma red local**.

---

## 📄 Referencia al Artículo Científico

Consulta el artículo detallado del proyecto:

📘 [CalderonArticuloCientificoJun232025.pdf](CalderonArticuloCientificoJun232025.pdf)

---

## 🎥 Video Tutorial

Demostración del sistema en acción:

📺 [DemostracionTutorialUso.mp4](DemostracionTutorialUso.mp4)

---

## 🤝 Agradecimientos

Al docente de **DESARROLLO DE APLICACIONES INTELIGENTES** por fomentar el desarrollo de soluciones aplicadas con impacto educativo y técnico.

---

## 📫 Contacto

¿Dudas o sugerencias?
✉️ [antoniocfbb17@gmail.com](mailto:antoniocfbb17@gmail.com) 
