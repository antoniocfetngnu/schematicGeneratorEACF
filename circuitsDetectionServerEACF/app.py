from flask import Flask, request, jsonify
import os
import cv2
import numpy as np
import base64
from ultralytics import YOLO
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import threading
import json
import random
from typing import List, Tuple, Optional
import time
from matplotlib.colors import hsv_to_rgb


app = Flask(__name__)

# Directorio base
base_output_dir = "preprocesamiento_individual"
os.makedirs(base_output_dir, exist_ok=True)

# Estado global
processing_state = {
    'is_processing': False,
    'detections_completed': False,
    'dip_selection_pending': False,
    'resistencias_df': None,
    'leds_df': None,
    'botones_df': None,
    'cristales_df': None,
    'cap_cers_df': None,
    'usonic_detections':None,
    'dips_df': None,
    'dip_detections': None,
    'dip_image_base64': None,
    'dip_selections': None,
    'unificado_df': None,
    'conexiones_df': None,
    'conexiones_limpio_df': None,  # Añade esta línea
    'schematic_data': []  # Añade esta línea
}

# Cargar modelos
modelo_componentes = YOLO("modelos/components.pt") #detector de componentes
modelo_patitas = YOLO("modelos/thinLeadsV2.pt")   # detector de patitas de componentes
modelo_dip = YOLO("modelos/pinsAndTipsSeg.pt") #detector de muescas y pines de DIPs
modelo_cables = YOLO("modelos/wireSegv2.pt")  #segmentador de cables
modelo_puntas = YOLO("modelos/crucesPuntasSegmentatorV2.pt") # #segmentador de puntas de cables
modelo_usonic = YOLO("modelos/usonicPinDetectorv2.pt") #segmentador de sensores ultrasónicos
# Cargar base de datos de DIPs
with open("componentes_dip_netlist.json", "r") as f:
    dip_netlist = json.load(f)
def visualizar_patitas_componente(recorte_contraste, masks, output_dir, component_type, idx):
    """
    Visualiza las patitas detectadas de un componente (resistencia o LED) y guarda la imagen en una subcarpeta.
    
    Args:
        recorte_contraste (np.ndarray): Imagen recortada del componente con contraste.
        masks (np.ndarray): Máscaras de las patitas detectadas.
        output_dir (str): Directorio base para guardar las imágenes.
        component_type (str): Tipo de componente ('resistencia' o 'led').
        idx (int): Índice del componente (para el nombre del archivo).
    """
    try:
        # Crear subcarpeta según el tipo de componente
        sub_dir = os.path.join(output_dir, component_type + "s")
        os.makedirs(sub_dir, exist_ok=True)
        
        # Convertir imagen a RGB
        img_debug = cv2.cvtColor(recorte_contraste, cv2.COLOR_BGR2RGB).copy()
        
        # Dibujar contornos de las patitas y etiquetas
        for j, mask in enumerate(masks):
            mask_bin = (mask > 0.5).astype(np.uint8)
            mask_resized = cv2.resize(mask_bin, (img_debug.shape[1], img_debug.shape[0]), 
                                    interpolation=cv2.INTER_NEAREST)
            contornos, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contornos:
                cv2.drawContours(img_debug, contornos, -1, (0, 255, 0), 2)  # Verde
                # Añadir etiqueta P{j} cerca del contorno
                x, y, _, _ = cv2.boundingRect(contornos[0])
                cv2.putText(img_debug, f"P{j}", (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Guardar imagen en la subcarpeta correspondiente
        output_path = os.path.join(sub_dir, f"{component_type}{idx + 1}.jpg")
        cv2.imwrite(output_path, cv2.cvtColor(img_debug, cv2.COLOR_RGB2BGR))
        print(f"🖼️ Imagen de {component_type} {idx + 1} guardada en: {output_path}")
        
    except Exception as e:
        print(f"❌ Error al visualizar {component_type} {idx}: {str(e)}")

# --- Funciones de procesamiento de pines ---
def load_dip_pin_counts(netlist_path="componentes_dip_netlist.json"):
    """
    Carga el conteo de pines y nombres de pines desde el archivo netlist o un diccionario.
    
    Args:
        netlist_path (str or dict): Ruta al archivo JSON de netlist o diccionario con los datos.
    
    Returns:
        tuple: (dip_pin_counts, pin_names_dict)
            - dip_pin_counts: Diccionario con el conteo de pines por lado para cada tipo de DIP.
            - pin_names_dict: Diccionario con nombres de pines por tipo de DIP.
    """
    try:
        # Si netlist_path es un diccionario, usarlo directamente
        if isinstance(netlist_path, dict):
            netlist = netlist_path
        else:
            # Si es una ruta, cargar el archivo
            with open(netlist_path, 'r') as f:
                netlist = json.load(f)
        
        dip_pin_counts = {}
        pin_names_dict = {}
        
        for dip_type, data in netlist.items():
            pin_count = data.get('pinSideCount', 7)  # Por defecto 7 pines por lado
            dip_pin_counts[dip_type] = pin_count
            # Copiar el diccionario de nombres de pines, excluyendo pinSideCount
            pin_names_dict[dip_type] = {k: v for k, v in data.items() if k != 'pinSideCount'}
        
        return dip_pin_counts, pin_names_dict
    
    except Exception as e:
        print(f"❌ Error al cargar netlist: {str(e)}")
        return {}, {}

def detectar_posicion_muesca(dip, box_width):
    if dip['muescas']:
        if len(dip['muescas']) > 1:
            print(f"⚠️ DIP {dip['id']} tiene {len(dip['muescas'])} muescas. Usando la primera.")
        muesca = dip['muescas'][0]
        centro_muesca_x = (muesca[0] + muesca[2]) / 2
        centro_dip_x = (dip['box'][0] + dip['box'][2]) / 2
        return "izquierda" if centro_muesca_x < centro_dip_x else "derecha"
    elif dip['points']:
        point = dip['points'][0]
        centro_point_x = (point[0] + point[2]) / 2
        centro_dip_x = (dip['box'][0] + dip['box'][2]) / 2
        return "izquierda" if centro_point_x < centro_dip_x else "derecha"
    else:
        print(f"⚠️ DIP {dip['id']} no tiene muescas ni points. Asumiendo muesca a la izquierda.")
        return "izquierda"

def renumerar_pines_esquema_U(pins_sup, pins_inf, posicion_muesca):
    """
    Renumera los pines de un integrado DIP siguiendo el estándar en U.
    
    Args:
        pins_sup: Lista de pines superiores
        pins_inf: Lista de pines inferiores  
        posicion_muesca: "izquierda" o "derecha" (posición de la muesca)
    
    Returns:
        dict: Mapeo de (fila, índice) -> número de pin
    """
    total_pines = len(pins_sup) + len(pins_inf)
    mapa_renumeracion = {}
    
    if posicion_muesca == "izquierda":
        # Muesca a la izquierda: empezar por pin 1 en inferior izquierda
        # Numeración: inferior de izquierda a derecha, luego superior de derecha a izquierda
        for nuevo_num, idx in enumerate(range(len(pins_inf)), 1):
            mapa_renumeracion[("inferior", idx)] = nuevo_num
        
        # Pines superiores: de derecha a izquierda
        idx_pins_sup = list(range(len(pins_sup)))
        idx_pins_sup.reverse()
        for nuevo_num, idx in enumerate(idx_pins_sup, len(pins_inf) + 1):
            mapa_renumeracion[("superior", idx)] = nuevo_num
            
    else:  # posicion_muesca == "derecha"
        # Muesca a la derecha: empezar por pin 1 en superior derecha
        # Numeración: superior de derecha a izquierda, luego inferior de izquierda a derecha
        idx_pins_sup = list(range(len(pins_sup)))
        idx_pins_sup.reverse()
        for nuevo_num, idx in enumerate(idx_pins_sup, 1):
            mapa_renumeracion[("superior", idx)] = nuevo_num
            
        # Pines inferiores: de izquierda a derecha
        for nuevo_num, idx in enumerate(range(len(pins_inf)), len(pins_sup) + 1):
            mapa_renumeracion[("inferior", idx)] = nuevo_num
    
    return mapa_renumeracion

def interpolar_pines_avanzado(sup_x: List[int], inf_x: List[int], box_width: int, 
                              esperados_por_lado: int, box_x_inicio: int = 0) -> Tuple[List[int], List[int], List[bool], List[bool]]:
    """
    Interpola pines de un DIP para ambas filas (superior e inferior), verifica distancias y 
    reconstruye basándose en una fila válida si es necesario.
    
    Args:
        sup_x: Lista de posiciones X de pines detectados en la fila superior (coordenadas globales).
        inf_x: Lista de posiciones X de pines detectados en la fila inferior (coordenadas globales).
        box_width: Ancho del bounding box del componente.
        esperados_por_lado: Número esperado de pines por fila.
        box_x_inicio: Coordenada X del inicio del bounding box (para convertir a coordenadas locales).
    
    Returns:
        Tuple[List[int], List[int], List[bool], List[bool]]: 
        (pines_superior_globales, pines_inferior_globales, sup_original, inf_original).
    """
    # Convertir a coordenadas locales
    sup_x_local = [x - box_x_inicio for x in sup_x]
    inf_x_local = [x - box_x_inicio for x in inf_x]
    
    # Eliminar pines duplicados o muy cercanos
    distancia_min = box_width * 0.05  # Umbral: 5% del ancho del bounding box
    sup_x_local = _eliminar_duplicados(sorted(sup_x_local), distancia_min)
    inf_x_local = _eliminar_duplicados(sorted(inf_x_local), distancia_min)
    
    # Generar distribución ideal
    distribucion_ideal = _calcular_distribucion_ideal(box_width, esperados_por_lado)
    
    # Caso: no hay pines detectados en ninguna fila
    if not sup_x_local and not inf_x_local:
        pines_local, es_original = _generar_distribucion_uniforme(box_width, esperados_por_lado)
        pines_global = [x + box_x_inicio for x in pines_local]
        return pines_global, pines_global, es_original, es_original
    
    # Interpolar pines para cada fila
    sup_pines, sup_original = _interpolar_pines(sup_x_local, distribucion_ideal, esperados_por_lado)
    inf_pines, inf_original = _interpolar_pines(inf_x_local, distribucion_ideal, esperados_por_lado)
    
    # Validar distancias entre pines
    distancia_promedio = box_width / (esperados_por_lado - 1) if esperados_por_lado > 1 else box_width
    sup_valido = _validar_distancias(sup_pines, distancia_promedio)
    inf_valido = _validar_distancias(inf_pines, distancia_promedio)
    
    # Reconstruir si una fila es inválida
    if sup_valido and not inf_valido:
        # Usar fila superior para reconstruir inferior (espejo)
        inf_pines = _reconstruir_desde_superior(sup_pines, box_width)
        inf_original = [False] * len(inf_pines)
    elif inf_valido and not sup_valido:
        # Usar fila inferior para reconstruir superior (espejo)
        sup_pines = _reconstruir_desde_inferior(inf_pines, box_width)
        sup_original = [False] * len(sup_pines)
    elif not sup_valido and not inf_valido:
        # Ambas filas inválidas: usar distribución uniforme
        sup_pines, sup_original = _generar_distribucion_uniforme(box_width, esperados_por_lado)
        inf_pines, inf_original = _generar_distribucion_uniforme(box_width, esperados_por_lado)
    
    # Ajustar límites y convertir a coordenadas globales
    sup_pines = _validar_y_ajustar_limites(sup_pines, box_width)
    inf_pines = _validar_y_ajustar_limites(inf_pines, box_width)
    sup_global = [x + box_x_inicio for x in sup_pines]
    inf_global = [x + box_x_inicio for x in inf_pines]
    
    return sup_global, inf_global, sup_original, inf_original


def _generar_distribucion_uniforme(box_width: int, num_pines: int) -> Tuple[List[int], List[bool]]:
    """Genera una distribución uniforme de pines."""
    margen = box_width * 0.12
    if num_pines == 1:
        return [int(box_width / 2)], [False]
    espacio_util = box_width - 2 * margen
    distancia = espacio_util / (num_pines - 1) if num_pines > 1 else 0
    pines = [int(margen + i * distancia) for i in range(num_pines)]
    return pines, [False] * num_pines


def _calcular_distribucion_ideal(box_width: int, num_pines: int) -> List[float]:
    """Calcula la distribución ideal de pines."""
    margen = box_width * 0.12
    if num_pines == 1:
        return [box_width / 2]
    espacio_util = box_width - 2 * margen
    distancia = espacio_util / (num_pines - 1) if num_pines > 1 else 0
    return [margen + i * distancia for i in range(num_pines)]


def _eliminar_duplicados(xs: List[int], distancia_min: float) -> List[int]:
    """Elimina pines duplicados o muy cercanos."""
    if not xs:
        return xs
    xs_filtrados = [xs[0]]
    for x in xs[1:]:
        if x - xs_filtrados[-1] >= distancia_min:
            xs_filtrados.append(x)
    return xs_filtrados


def _interpolar_pines(xs_detectados: List[int], distribucion_ideal: List[float], 
                      num_pines: int) -> Tuple[List[int], List[bool]]:
    """Interpolar pines faltantes usando la distribución ideal."""
    pines_finales = []
    es_original = []
    
    # Mapear pines detectados a las posiciones ideales más cercanas
    mapeo = [None] * num_pines
    for i, x_det in enumerate(xs_detectados):
        distancias = [abs(x_det - x_ideal) for x_ideal in distribucion_ideal]
        idx_ideal = distancias.index(min(distancias))
        if mapeo[idx_ideal] is None:  # Evitar sobrescribir
            mapeo[idx_ideal] = x_det
    
    # Generar pines finales
    for i in range(num_pines):
        if mapeo[i] is not None:
            pines_finales.append(mapeo[i])
            es_original.append(True)
        else:
            pines_finales.append(int(round(distribucion_ideal[i])))
            es_original.append(False)
    
    return sorted(pines_finales), es_original


def _validar_distancias(pines: List[int], distancia_promedio: float) -> bool:
    """Verifica si las distancias entre pines son razonables."""
    if len(pines) < 2:
        return True
    distancias = [pines[i+1] - pines[i] for i in range(len(pines)-1)]
    umbral_min = distancia_promedio * 0.7  # Tolerancia: ±30%
    umbral_max = distancia_promedio * 1.3
    umbral_duplicado = distancia_promedio * 0.2  # Pines muy cercanos
    return all(umbral_min <= d <= umbral_max for d in distancias) and min(distancias) > umbral_duplicado


def _reconstruir_desde_superior(sup_pines: List[int], box_width: int) -> List[int]:
    """Reconstruye pines inferiores usando los superiores como espejo."""
    mitad_box = box_width / 2
    return [int(2 * mitad_box - x) for x in sup_pines[::-1]]  # Reflejar en espejo


def _reconstruir_desde_inferior(inf_pines: List[int], box_width: int) -> List[int]:
    """Reconstruye pines superiores usando los inferiores como espejo."""
    mitad_box = box_width / 2
    return [int(2 * mitad_box - x) for x in inf_pines[::-1]]  # Reflejar en espejo


def _validar_y_ajustar_limites(pines: List[int], box_width: int) -> List[int]:
    """Ajusta pines para respetar los límites del bounding box."""
    margen = box_width * 0.05
    return [max(int(margen), min(int(box_width - margen), pin)) for pin in pines]

def interpolar_pines_simple(xs_detectados, box_width, esperados_por_lado):
    if not xs_detectados:
        margen = box_width * 0.15
        distancia = (box_width - 2 * margen) / (esperados_por_lado - 1) if esperados_por_lado > 1 else box_width / 2
        pins = [int(margen + i * distancia) for i in range(esperados_por_lado)]
        es_original = [False] * esperados_por_lado
        return pins, es_original
    
    xs = sorted(xs_detectados)
    
    if len(xs) >= esperados_por_lado:
        return xs[:esperados_por_lado], [True] * esperados_por_lado
    
    if len(xs) >= 2:
        distancias = [xs[i+1] - xs[i] for i in range(len(xs) - 1)]
        distancia_prom = sum(distancias) / len(distancias)
    else:
        distancia_prom = box_width / (esperados_por_lado + 1)
    
    nuevos_pins = xs.copy()
    es_original = [True] * len(xs)
    
    if len(xs) >= 2:
        for i in range(len(xs) - 1):
            distancia_actual = xs[i+1] - xs[i]
            if distancia_actual > distancia_prom * 1.5:
                pines_a_agregar = round(distancia_actual / distancia_prom) - 1
                for j in range(1, pines_a_agregar + 1):
                    pos_x = xs[i] + j * (distancia_actual / (pines_a_agregar + 1))
                    nuevos_pins.append(int(pos_x))
                    es_original.append(False)
    
    primer_x = min(nuevos_pins)
    distancia_izquierda = primer_x
    if distancia_izquierda > distancia_prom * 1.5:
        pines_izquierda = min(
            round(distancia_izquierda / distancia_prom),
            esperados_por_lado - len(nuevos_pins)
        )
        for i in range(1, pines_izquierda + 1):
            pos_x = max(0, primer_x - i * distancia_prom)
            nuevos_pins.append(int(pos_x))
            es_original.append(False)
    
    ultimo_x = max(nuevos_pins)
    distancia_derecha = box_width - ultimo_x
    if distancia_derecha > distancia_prom * 1.5:
        pines_derecha = min(
            round(distancia_derecha / distancia_prom),
            esperados_por_lado - len(nuevos_pins)
        )
        for i in range(1, pines_derecha + 1):
            pos_x = min(box_width, ultimo_x + i * distancia_prom)
            nuevos_pins.append(int(pos_x))
            es_original.append(False)
    
    pins_ordenados = sorted(zip(nuevos_pins, es_original))
    
    while len(pins_ordenados) < esperados_por_lado:
        mayor_espacio = 0
        pos_insercion = 0
        for i in range(len(pins_ordenados) - 1):
            espacio = pins_ordenados[i+1][0] - pins_ordenados[i][0]
            if espacio > mayor_espacio:
                mayor_espacio = espacio
                pos_insercion = i
        nuevo_pin = pins_ordenados[pos_insercion][0] + mayor_espacio // 2
        pins_ordenados.insert(pos_insercion + 1, (nuevo_pin, False))
    
    pins_ordenados = pins_ordenados[:esperados_por_lado]
    xs_final, es_orig_final = zip(*pins_ordenados)
    return list(xs_final), list(es_orig_final)
def interpolar_pines_optimista(sup_x: list[int], inf_x: list[int], x1: int, x2: int, esperados_por_lado: int) -> tuple[list[int], list[int], list[bool], list[bool]]:
    """
    Interpola pines de un DIP para ambas filas (superior e inferior) usando un enfoque iterativo basado en la 
    distancia mínima entre pines consecutivos dentro de la misma fila. Solo la fila con más pines detectados
    se analiza e interpola; la otra fila se rellena como espejo de la base, respetando los límites globales x1 y x2.
    
    Args:
        sup_x: Lista de posiciones X de pines detectados en la fila superior (coordenadas globales).
        inf_x: Lista de posiciones X de pines detectados en la fila inferior (coordenadas globales).
        x1: Coordenada X del inicio del bounding box (límite izquierdo global).
        x2: Coordenada X del fin del bounding box (límite derecho global).
        esperados_por_lado: Número esperado de pines por fila.
    
    Returns:
        Tuple[List[int], List[int], List[bool], List[bool]]: 
        (pines_superior_globales, pines_inferior_globales, sup_original, inf_original).
    """
    # Validar que x1 < x2
    if x1 >= x2:
        raise ValueError("x1 debe ser menor que x2.")
    
    box_width = x2 - x1
    margen = box_width * 0.10  # Margen relativo al ancho del bounding box
    
    # Calcular distancia mínima dentro de cada fila
    def calcular_distancia_min(fila: list[int]) -> float:
        if len(fila) < 2:
            raise ValueError("Se necesitan al menos 2 pines consecutivos en una fila.")
        return min(fila[i+1] - fila[i] for i in range(len(fila)-1))
    
    dist_min_sup = calcular_distancia_min(sup_x) if len(sup_x) > 1 else float('inf')
    dist_min_inf = calcular_distancia_min(inf_x) if len(inf_x) > 1 else float('inf')
    distancia_min = min(dist_min_sup, dist_min_inf)
    if distancia_min == float('inf'):
        raise ValueError("No se encontraron al menos 2 pines consecutivos en ninguna fila.")
    
    # Elegir fila con más pines como base
    use_sup = len(sup_x) >= len(inf_x)
    base_x = sup_x if use_sup else inf_x
    
    # Inicializar pines y marcas (los pines detectados no se mueven)
    base_pins = base_x.copy()
    base_original = [True] * len(base_x)
    
    # Paso iterativo: rellenar huecos entre pines
    while len(base_pins) < esperados_por_lado:
        base_pins_sorted = sorted(base_pins)
        nuevo_pin_agregado = False
        
        # Revisar huecos entre pines
        for i in range(len(base_pins_sorted) - 1):
            distancia_actual = base_pins_sorted[i+1] - base_pins_sorted[i]
            if distancia_actual > distancia_min and not nuevo_pin_agregado:
                pos_x = base_pins_sorted[i] + distancia_min
                if pos_x < base_pins_sorted[i+1] and pos_x not in base_pins and pos_x >= x1 + margen and pos_x <= x2 - margen:
                    base_pins.append(int(pos_x))
                    base_original.append(False)
                    nuevo_pin_agregado = True
        
        # Iteración explícita desde los extremos
        if not nuevo_pin_agregado and len(base_pins) < esperados_por_lado:
            primer_x = min(base_pins_sorted)
            ultimo_x = max(base_pins_sorted)
            
            # Extremo izquierdo
            if primer_x - distancia_min >= x1 + margen and not nuevo_pin_agregado:
                pos_x = primer_x - distancia_min
                if pos_x not in base_pins and pos_x >= x1 + margen:
                    base_pins.insert(0, int(pos_x))
                    base_original.insert(0, False)
                    nuevo_pin_agregado = True
            
            # Extremo derecho
            if ultimo_x + distancia_min <= x2 - margen and not nuevo_pin_agregado:
                pos_x = ultimo_x + distancia_min
                if pos_x not in base_pins and pos_x <= x2 - margen:
                    base_pins.append(int(pos_x))
                    base_original.append(False)
                    nuevo_pin_agregado = True
        
        # Si no se pudo añadir ningún pin, salir para evitar bucles infinitos
        if not nuevo_pin_agregado:
            break
    
    # Ajustar a esperados_por_lado si hay más pines (truncar) o menos (rellenar uniformemente si es necesario)
    base_pins = sorted(base_pins)
    if len(base_pins) > esperados_por_lado:
        base_pins = base_pins[:esperados_por_lado]
        base_original = base_original[:esperados_por_lado]
    elif len(base_pins) < esperados_por_lado:
        while len(base_pins) < esperados_por_lado:
            ultimo_x = base_pins[-1]
            pos_x = min(x2 - margen, ultimo_x + distancia_min)
            if pos_x not in base_pins and pos_x >= x1 + margen and pos_x <= x2 - margen:
                base_pins.append(int(pos_x))
                base_original.append(False)
    
    # Reconstruir la otra fila como espejo exacto de la base, ignorando pines detectados en la otra fila
    other_pins = []
    other_original = [False] * esperados_por_lado  # Todos serán interpolados como espejo
    mitad_box = (x1 + x2) / 2
    for i in range(esperados_por_lado):
        if i < len(base_pins):
            espejo_x = int(2 * mitad_box - base_pins[esperados_por_lado - 1 - i])
            other_pins.append(espejo_x)
    
    # Asegurar que other_pins respete los límites globales
    other_pins = [max(x1 + margen, min(x2 - margen, x)) for x in other_pins]
    
    # Asignar resultados según fila usada
    if use_sup:
        return base_pins, other_pins, base_original, other_original
    else:
        return other_pins, base_pins, other_original, base_original
    
# def interpolacion_propia(sup_x: list[int], inf_x: list[int], x1: int, x2: int, esperados_por_lado: int) -> tuple[list[int], list[int], list[bool], list[bool]]:
#     print(f"📌 Debugging - Límites globales: x1 = {x1}, x2 = {x2}")
#     pins_sup = sup_x.copy()
#     pins_inf = inf_x.copy()
#     sup_orig = [True] * len(sup_x)
#     inf_orig = [True] * len(inf_x)

#     if(x1>x2):
#         limiteIzquierdo = x2
#         limiteDerecho = x1
#     else:
#         limiteIzquierdo = x1
#         limiteDerecho = x2
#     def calcular_distancia_min(fila: list[int]) -> float:
#         if len(fila) < 2:
#             raise ValueError("Se necesitan al menos 2 pines consecutivos en una fila.")
#         return min(fila[i+1] - fila[i] for i in range(len(fila)-1))
   
        
#     # contar que fila tiene mas pines, pins_sup o pins_inf
#     if len(pins_sup) >= len(pins_inf):
#         fila_base = pins_sup
#         fila_otra = pins_inf
#         orig_base = sup_orig
#         orig_otra = inf_orig
#     else:
#         fila_base = pins_inf
#         fila_otra = pins_sup    
#         orig_base = inf_orig
#         orig_otra = sup_orig

#     fila_base = sorted(fila_base)

#     minima_distancia = calcular_distancia_min(fila_base);

#     #ordenar pines en fila base de menor a mayor
    
#     fila_otra = sorted(fila_otra)
    
#     print(f"📌 Debugging - Fila base (pines): {fila_base}")
#     while True:
#         pin_added = False
#     #Verificar si hay una distancia superior a 1.4 de la distancia minima entre pines consecutivos
#         for i in range(len(fila_base)-1):
#             if fila_base[i+1] - fila_base[i] > minima_distancia * 1.4:
#                 print(f"⚠️ Distancia entre pines {fila_base[i]} y {fila_base[i+1]} es mayor a 1.4 veces la distancia mínima ({minima_distancia}).")
#                 pos_x = fila_base[i] + minima_distancia
#                 if pos_x < fila_base[i+1] and pos_x >= limiteIzquierdo and pos_x <= limiteDerecho:
#                     fila_base.append(int(pos_x))
#                     orig_base.append(False)  # Marcar como interpolado
#                     fila_base = sorted(fila_base)
#                     pin_added = True
#                     break
#         if not pin_added:
#             break  # Salir del while si no se añadió ningún pin
#     # if length fila_base < esperados_por_lado:
#     contadorBreak = 0
#     while (len(fila_base) < esperados_por_lado):
#         agregoPin = False
#         print(f"📌 Debugging - Pines en fila base: {len(fila_base)} (esperados: {esperados_por_lado})")
#         # tomar el primer pin y el ultimo pin de la fila base
#         primer_pin = fila_base[0]
#         ultimo_pin = fila_base[-1]
#         # Calcular el espacio disponible a la izquierda y derecha
#         espacio_izquierda = primer_pin - limiteIzquierdo
#         espacio_derecha = limiteDerecho - ultimo_pin
#         print(f"📌 Debugging - Espacio disponible: Izquierda = {espacio_izquierda}, Derecha = {espacio_derecha}")
#         if(espacio_izquierda > espacio_derecha and espacio_izquierda > minima_distancia * 1.4):
#             print(f"📌 Debugging - Agregando pines a la izquierda")
#             # Agregar un pin a la izquierda
#             pos_x = primer_pin - minima_distancia
#             if pos_x >= limiteIzquierdo and pos_x <= limiteDerecho:
#                 fila_base.insert(0, int(pos_x))
#                 orig_base.insert(0, False)
#                 agregoPin = True
            
    
#         elif(espacio_derecha > espacio_izquierda and espacio_derecha > minima_distancia * 1.4):
#             print(f"📌 Debugging - Agregando pines a la derecha")
#             # Agregar un pin a la derecha
#             pos_x = ultimo_pin + minima_distancia
#             if pos_x >= limiteIzquierdo and pos_x <= limiteDerecho:
#                 fila_base.append(int(pos_x))
#                 orig_base.append(False)
#                 agregoPin = True

#         if(agregoPin == False):
#             contadorBreak += 1
#         if(contadorBreak > 10):
#             print(f"⚠️ Demasiados intentos de agregar pines, saliendo del bucle.")
#             break
#         contadorBreak += 1
            
#     # Actualizar pins_sup y pins_inf con los resultados
#     fila_base = sorted(fila_base)
#     fila_otra.clear()  # Limpiar la fila opuesta para reconstruirla
#     fila_otra = fila_base.copy()  # Copiar la fila base para la otra fila   
    
#     # Corrección del return para reflejar los cambios
#     if len(pins_sup) >= len(pins_inf):  # Si pins_sup era la fila base original
#         pins_sup = fila_base
#         pins_inf = fila_otra
#         sup_orig = orig_base
#         inf_orig = orig_otra  # Ajustar orig_otra si es necesario (todos False tras clear)
#     else:  # Si pins_inf era la fila base original
#         pins_inf = fila_base
#         pins_sup = fila_otra
#         inf_orig = orig_base
#         sup_orig = orig_otra  # Ajustar orig_otra si es necesario

#     return pins_sup, pins_inf, sup_orig, inf_orig
def interpolacion_propia(sup_x: list[int], inf_x: list[int], x1: int, x2: int, esperados_por_lado: int) -> tuple[list[int], list[int], list[bool], list[bool]]:
    print(f"📌 Debugging - Límites globales: x1 = {x1}, x2 = {x2}")
    pins_sup = sup_x.copy()
    pins_inf = inf_x.copy()
    sup_orig = [True] * len(sup_x)
    inf_orig = [True] * len(inf_x)

    if(x1>x2):
        limiteIzquierdo = x2
        limiteDerecho = x1
    else:
        limiteIzquierdo = x1
        limiteDerecho = x2
    def calcular_distancia_min(fila: list[int]) -> float:
        if len(fila) < 2:
            raise ValueError("Se necesitan al menos 2 pines consecutivos en una fila.")
        return min(fila[i+1] - fila[i] for i in range(len(fila)-1))
        
    # contar que fila tiene mas pines, pins_sup o pins_inf
    if len(pins_sup) >= len(pins_inf):
        fila_base = pins_sup
        fila_otra = pins_inf
        orig_base = sup_orig
        orig_otra = inf_orig
    else:
        fila_base = pins_inf
        fila_otra = pins_sup    
        orig_base = inf_orig
        orig_otra = sup_orig
    
    fila_base = sorted(fila_base)
    fila_otra = sorted(fila_otra)    

    minima_distancia = calcular_distancia_min(fila_base);

    #ordenar pines en fila base de menor a mayor
    
    
    print(f"📌 Debugging - Fila base (pines): {fila_base}")
    while True:
        pin_added = False
    #Verificar si hay una distancia superior a 1.4 de la distancia minima entre pines consecutivos
        for i in range(len(fila_base)-1):
            if fila_base[i+1] - fila_base[i] > minima_distancia * 1.6:
                print(f"⚠️ Distancia entre pines {fila_base[i]} y {fila_base[i+1]} es mayor a 1.4 veces la distancia mínima ({minima_distancia}).")
                pos_x = fila_base[i] + minima_distancia
                if pos_x < fila_base[i+1] and pos_x >= limiteIzquierdo and pos_x <= limiteDerecho:
                    fila_base.append(int(pos_x))
                    orig_base.append(False)  # Marcar como interpolado
                    fila_base = sorted(fila_base)
                    pin_added = True
                    print(f"Pin añadido! Nuevos pines! (Límite Izquierdo: {limiteIzquierdo}, Límite Derecho: {limiteDerecho})")
                    print(f"📌 Debugging - Pines actualizados: {fila_base}")
                    break
        if not pin_added:
            break  # Salir del while si no se añadió ningún pin
    # if length fila_base < esperados_por_lado:
    contadorBreak = 0
    while (len(fila_base) < esperados_por_lado):
        agregoPin = False
        print(f"📌 Debugging - Pines en fila base: {len(fila_base)} (esperados: {esperados_por_lado})")
        # tomar el primer pin y el ultimo pin de la fila base
        primer_pin = fila_base[0]
        ultimo_pin = fila_base[-1]
        # Calcular el espacio disponible a la izquierda y derecha
        espacio_izquierda = primer_pin - limiteIzquierdo
        espacio_derecha = limiteDerecho - ultimo_pin
        print(f"📌 Debugging - Espacio disponible: Izquierda = {espacio_izquierda}, Derecha = {espacio_derecha}")
        if(espacio_izquierda > espacio_derecha and espacio_izquierda > minima_distancia * 1.4):
            print(f"📌 Debugging - Agregando pines a la izquierda")
            # Agregar un pin a la izquierda
            pos_x = primer_pin - minima_distancia
            if pos_x >= limiteIzquierdo and pos_x <= limiteDerecho:
                fila_base.insert(0, int(pos_x))
                orig_base.insert(0, False)
                agregoPin = True
                print(f"Pin añadido! Nuevos pines! (Límite Izquierdo: {limiteIzquierdo}, Límite Derecho: {limiteDerecho})")
                print(f"📌 Debugging - Pines actualizados: {fila_base}")
            
    
        elif(espacio_derecha > espacio_izquierda and espacio_derecha > minima_distancia * 1.4):
            print(f"📌 Debugging - Agregando pines a la derecha")
            # Agregar un pin a la derecha
            pos_x = ultimo_pin + minima_distancia
            if pos_x >= limiteIzquierdo and pos_x <= limiteDerecho:
                fila_base.append(int(pos_x))
                orig_base.append(False)
                agregoPin = True
                print(f"Pin añadido! Nuevos pines! (Límite Izquierdo: {limiteIzquierdo}, Límite Derecho: {limiteDerecho})")
                print(f"📌 Debugging - Pines actualizados: {fila_base}")

        if(agregoPin == False):
            contadorBreak += 1
        if(contadorBreak > 10):
            print(f"⚠️ Demasiados intentos de agregar pines, saliendo del bucle.")
            break
        contadorBreak += 1
            
    # Actualizar pins_sup y pins_inf con los resultados
    fila_base = sorted(fila_base)
    fila_otra.clear()  # Limpiar la fila opuesta para reconstruirla
    fila_otra = fila_base.copy()  # Copiar la fila base para la otra fila   
    
    # Corrección del return para reflejar los cambios
    if len(pins_sup) >= len(pins_inf):  # Si pins_sup era la fila base original
        pins_sup = fila_base
        pins_inf = fila_otra
        sup_orig = orig_base
        inf_orig = orig_otra  # Ajustar orig_otra si es necesario (todos False tras clear)
    else:  # Si pins_inf era la fila base original
        pins_inf = fila_base
        pins_sup = fila_otra
        inf_orig = orig_base
        sup_orig = orig_otra  # Ajustar orig_otra si es necesario

    return pins_sup, pins_inf, sup_orig, inf_orig

    
def procesar_dips_simple(dip_detecciones_detalladas, dip_types, dip_pin_counts):
    for dip in dip_detecciones_detalladas:
        idx = dip['id']
        box = dip['box']
        x1 = box[0]  # Límite izquierdo del bounding box
        x2 = box[2]  # Límite derecho del bounding box
        box_width = box[2] - box[0]
        box_x_inicio = box[0]  # ← AGREGAR ESTA LÍNEA

        print(f"📌 Debugging - DIP {idx}: x1 = {x1}, x2 = {x2}")
        
        dip_type = dip_types.get(idx, "unknown")
        esperados_por_lado = dip_pin_counts.get(dip_type, 7)
        print(f"📌 Procesando DIP {idx} (tipo: {dip_type}, pines por lado: {esperados_por_lado})")
        
        sup_x = [p['x'] for p in dip['pins_superior']]
        inf_x = [p['x'] for p in dip['pins_inferior']]

        print(f"📌 Debugging - DIP {idx}: Pines superiores (sup_x) = {sup_x}")
        print(f"📌 Debugging - DIP {idx}: Pines inferiores (inf_x) = {inf_x}")
        
        # # ← MODIFICAR ESTAS LLAMADAS para incluir box_x_inicio
        # pins_sup_interp, sup_original = interpolar_pines_simple(sup_x, box_width, esperados_por_lado)
        # pins_inf_interp, inf_original = interpolar_pines_simple(inf_x, box_width, esperados_por_lado)

        # pins_sup, pins_inf, sup_orig, inf_orig = interpolar_pines_optimista(sup_x, inf_x, x1, x2, esperados_por_lado)
        pins_sup_interp, pins_inf_interp, sup_original, inf_original = interpolacion_propia(sup_x, inf_x, x1, x2, esperados_por_lado)
        
        posicion_muesca = detectar_posicion_muesca(dip, box_width)
        mapa_renumeracion = renumerar_pines_esquema_U(pins_sup_interp, pins_inf_interp, posicion_muesca)

        # mapa_renumeracion = renumerar_pines_esquema_U(pins_sup, pins_inf, posicion_muesca)
        
        dip['pins_sup_interp'] = pins_sup_interp
        dip['pins_inf_interp'] = pins_inf_interp
        dip['sup_original'] = sup_original
        dip['inf_original'] = inf_original
        dip['type'] = dip_type
        dip['posicion_muesca'] = posicion_muesca
        dip['mapa_renumeracion'] = mapa_renumeracion
        
        sup_orig_count = sum(sup_original)
        inf_orig_count = sum(inf_original)
        print(f"  - Superior: {sup_orig_count} detectados, {len(sup_original)-sup_orig_count} interpolados")
        print(f"  - Inferior: {inf_orig_count} detectados, {len(inf_original)-inf_orig_count} interpolados")
    
    print("\n📊 Resumen de interpolación por DIP:")
    for dip in dip_detecciones_detalladas:
        sup_orig = sum(dip.get('sup_original', []))
        sup_total = len(dip.get('pins_sup_interp', []))
        inf_orig = sum(dip.get('inf_original', []))
        inf_total = len(dip.get('pins_inf_interp', []))
        print(f"- DIP {dip['id']} (tipo: {dip['type']}, Muesca: {dip['posicion_muesca']}):")
        print(f"   - Pines superior: {sup_orig} detectados + {sup_total - sup_orig} interpolados = {sup_total} total")
        print(f"   - Pines inferior: {inf_orig} detectados + {inf_total - inf_orig} interpolados = {inf_total} total")
    
    return dip_detecciones_detalladas

def crear_dataframe_pines_simple(dips_datos, pin_names_dict):
    """
    Crea un DataFrame de pines para DIPs a partir de dips_datos, incluyendo tipo_componente
    y nombres de pines específicos (A1, GND, etc.) desde pin_names_dict.
    
    Args:
        dips_datos (list): Lista de DIPs procesados con id, type, pins_sup_interp, pins_inf_interp.
        pin_names_dict (dict): Diccionario con nombres de pines por tipo de DIP.
    
    Returns:
        pd.DataFrame: DataFrame con pines de DIPs.
    """
    filas = []
    for dip in dips_datos:
        dip_id = dip['id']
        dip_type = dip['type']  # 7400, 7408, 7432, etc.
        posicion_muesca = dip['posicion_muesca']
        mapa_renumeracion = dip['mapa_renumeracion']
        # Obtener nombres de pines para este tipo de DIP
        pin_names = pin_names_dict.get(dip_type, {})
        
        # Pines superiores
        for i, x in enumerate(dip.get('pins_sup_interp', [])):
            es_original = dip.get('sup_original', [])[i] if i < len(dip.get('sup_original', [])) else False
            y_coord = dip['prom_y_sup'] if dip['prom_y_sup'] is not None else None
            numero_pin_u = mapa_renumeracion.get(("superior", i), i+1)
            # Usar str(numero_pin_u) para coincidir con las claves en pin_names
            pin_name = pin_names.get(str(numero_pin_u), f"Pin{numero_pin_u}")
            filas.append({
                'dip_id': dip_id,
                'tipo_componente': dip_type,
                'pin_id': i + 1,
                'pin_id_u': numero_pin_u,
                'pin_name': pin_name,
                'fila': 'superior',
                'x': int(x),
                'y': int(y_coord) if y_coord is not None else None,
                'detectado': bool(es_original),
                'interpolado': not bool(es_original),
                'posicion_muesca': posicion_muesca
            })
        
        # Pines inferiores
        for i, x in enumerate(dip.get('pins_inf_interp', [])):
            es_original = dip.get('inf_original', [])[i] if i < len(dip.get('inf_original', [])) else False
            y_coord = dip['prom_y_inf'] if dip['prom_y_inf'] is not None else None
            numero_pin_u = mapa_renumeracion.get(("inferior", i), i+1)
            # Usar str(numero_pin_u) para coincidir con las claves en pin_names
            pin_name = pin_names.get(str(numero_pin_u), f"Pin{numero_pin_u}")
            filas.append({
                'dip_id': dip_id,
                'tipo_componente': dip_type,
                'pin_id': i + 1,
                'pin_id_u': numero_pin_u,
                'pin_name': pin_name,
                'fila': 'inferior',
                'x': int(x),
                'y': int(y_coord) if y_coord is not None else None,
                'detectado': bool(es_original),
                'interpolado': not bool(es_original),
                'posicion_muesca': posicion_muesca
            })
    
    df = pd.DataFrame(filas) if filas else pd.DataFrame(columns=[
        'dip_id', 'tipo_componente', 'pin_id', 'pin_id_u', 'pin_name', 'fila', 'x', 'y', 'detectado', 'interpolado', 'posicion_muesca'
    ])
    return df

def continuar_procesar_dips():
    """
    Procesa las selecciones de DIPs, genera un DataFrame de pines con tipos específicos
    y unifica con otros componentes. Llama a funciones posteriores para generar conexiones
    y JSON.
    """
    global processing_state, dip_netlist, base_output_dir
    
    try:
        dip_detecciones = processing_state.get('dip_detections', [])
        dip_selections = processing_state.get('dip_selections', [])
        
        if not dip_detecciones or not dip_selections:
            print("❌ Error: No hay detecciones o selecciones disponibles")
            raise ValueError("Detecciones o selecciones vacías")
        
        detected_ids = {dip["id"] for dip in dip_detecciones}
        selection_ids = {sel["id"] for sel in dip_selections}
        if detected_ids != selection_ids:
            print(f"❌ Error: IDs de selecciones {selection_ids} no coinciden con detecciones {detected_ids}")
            raise ValueError("IDs no coinciden")
        
        # Filtrar DIPs válidos (con pines)
        dip_detecciones = [
            dip for dip in dip_detecciones
            if len(dip['pins_superior']) > 0 or len(dip['pins_inferior']) > 0
        ]
        print(f"✅ DIPs válidos: {len(dip_detecciones)}")
        
        # Mapear tipos específicos desde selecciones
        dip_types = {sel['id']: sel['type'] for sel in dip_selections}
        print(f"📥 Tipos de DIP seleccionados: {dip_types}")
        
        # Cargar conteo de pines y nombres
        dip_pin_counts, pin_names_dict = load_dip_pin_counts(dip_netlist)
        
        # Procesar DIPs con tipos específicos
        dips_procesados = procesar_dips_simple(dip_detecciones, dip_types, dip_pin_counts)
        
        # Crear DataFrame de pines con tipos específicos
        df_pines = crear_dataframe_pines_simple(dips_procesados, pin_names_dict)
        
        # Crear DataFrame de DIPs con información adicional
        dips_df = pd.DataFrame([
            {
                "id": dip["id"],
                "type": dip["type"],  # Usar tipo específico (7404, 7408, etc.)
                "box": dip["box"],
                "muescas": len(dip["muescas"]),
                "points": len(dip["points"]),
                "x_pins_sup": [p["x"] for p in dip["pins_superior"]],
                "x_pins_inf": [p["x"] for p in dip["pins_inferior"]],
                "prom_y_sup": dip["prom_y_sup"],
                "prom_y_inf": dip["prom_y_inf"],
                "interpolated": True,
                "posicion_muesca": dip["posicion_muesca"]
            }
            for dip in dips_procesados
        ])
        
        # Guardar df_pines como dips_df
        processing_state['dips_df'] = df_pines
        processing_state['detections_completed'] = True
        processing_state['is_processing'] = False
        processing_state['dip_selection_pending'] = False
        
        print("✅ Interpolación de pines completada.")
        print("📊 DataFrame de DIPs:")
        print(dips_df.to_string())
        print("📊 DataFrame final de pines:")
        print(df_pines.to_string())

        # Generar imagen con todos los componentes
        imagen_base_path = os.path.join(base_output_dir, "3000x3000.jpg")
        base_img = cv2.imread(imagen_base_path)
        generar_imagen_componentes(
            base_img,
            processing_state['resistencias_df'],
            processing_state['leds_df'],
            processing_state['botones_df'],
            processing_state['cables_df'],
            processing_state['cristales_df'],
            processing_state['cap_cers_df'],
            processing_state['usonic_detections'],
            processing_state['dips_df']
        )

        # Unificar DataFrames
        unificar_dataframes(
            processing_state['resistencias_df'],
            processing_state['leds_df'],
            processing_state['botones_df'],
            processing_state['cables_df'],
            processing_state['usonic_detections'],
            processing_state['cristales_df'],
            processing_state['cap_cers_df'],
            processing_state['dips_df']
        )

        # Generar conexiones
        generar_conexiones_carriles(processing_state['unificado_df'])

        # Visualizar conexiones
        visualizar_conexiones(processing_state['conexiones_df'])

        # Limpiar conexiones
        df_limpio = limpiar_conexiones(processing_state['conexiones_df'])
        
        # Generar JSON
        # generar_json_schematic(df_limpio)
        generar_json_desde_df(df_limpio)
    
    except Exception as e:
        print(f"❌ Error al procesar DIPs: {str(e)}")
        processing_state['is_processing'] = False
        processing_state['dip_selection_pending'] = False
def visualizar_cables_y_puntas(base_img, masks_cables, masks_tips, boxes_tips, output_dir):
    """
    Genera una imagen de debugging con las segmentaciones de cables y puntas.
    
    Args:
        base_img (np.ndarray): Imagen base de cables (1024x1024).
        masks_cables (list): Lista de máscaras de cables (tensores).
        masks_tips (list): Lista de máscaras de puntas (tensores).
        boxes_tips (list): Lista de bounding boxes de puntas (tensores).
        output_dir (str): Directorio para guardar la imagen.
    """
    try:
        # Convertir a RGB y crear copia
        img_debug = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB).copy()
        
        # Dibujar cables (contornos verdes)
        for idx, mask in enumerate(masks_cables):
            mask_bin = (mask.cpu().numpy() > 0.5).astype(np.uint8)
            mask_resized = cv2.resize(mask_bin, (img_debug.shape[1], img_debug.shape[0]), 
                                    interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img_debug, contours, -1, (0, 255, 0), 2)  # Verde
            # Añadir etiqueta
            if contours:
                x, y, _, _ = cv2.boundingRect(contours[0])
                cv2.putText(img_debug, f"C{idx}", (x, y - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Dibujar puntas (bounding boxes rojos y círculos)
        for idx, (box, mask) in enumerate(zip(boxes_tips, masks_tips)):
            x1, y1, x2, y2 = box.cpu().numpy().astype(int)
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, img_debug.shape[1]), min(y2, img_debug.shape[0])
            cv2.rectangle(img_debug, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Rojo
            # Centro del box
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(img_debug, (cx, cy), 5, (255, 0, 0), -1)
            # Etiqueta
            cv2.putText(img_debug, f"T{idx}", (x1, y1 - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        # Guardar imagen
        output_path = os.path.join(output_dir, "detecciones_cables_y_puntas.jpg")
        cv2.imwrite(output_path, cv2.cvtColor(img_debug, cv2.COLOR_RGB2BGR))
        print(f"🖼️ Imagen de debugging de cables y puntas guardada en: {output_path}")
        
    except Exception as e:
        print(f"❌ Error al generar imagen de debugging de cables y puntas: {str(e)}")

def mejorar_mascara_cable(img_base, mascara_original, factor_tolerancia=1.5, tolerancia_base=15, tolerancia_max=35, zona_cercana=30, zona_amplia=80, umbral_mejora=0.3):
    """
    Mejora la máscara de UN cable específico usando análisis de color con búsqueda adaptativa.
    
    Args:
        img_base: Imagen base
        mascara_original: Máscara original del cable
        factor_tolerancia: Factor multiplicador para std (default: 1.5)
        tolerancia_base: Tolerancia mínima en píxeles (default: 15) 
        tolerancia_max: Tolerancia máxima en píxeles (default: 35)
        zona_cercana: Radio de búsqueda inicial (default: 30)
        zona_amplia: Radio de búsqueda ampliada (default: 80)
        umbral_mejora: Umbral para ampliar búsqueda (default: 0.3 = 30%)
    """
    try:
        # 1. Extraer colores solo de esta máscara específica
        puntos_cable = np.where(mascara_original > 0)
        if len(puntos_cable[0]) == 0:
            return mascara_original
        
        colores_cable = img_base[puntos_cable]
        
        # 2. Análisis de color
        color_medio = np.mean(colores_cable, axis=0).astype(np.uint8)
        color_std = np.std(colores_cable, axis=0).astype(np.uint8)
        
        # 3. PARÁMETROS AJUSTABLES
        tolerancia = np.maximum(color_std * factor_tolerancia, tolerancia_base)
        tolerancia = np.minimum(tolerancia, tolerancia_max)
        
        color_min = np.clip(color_medio - tolerancia, 0, 255).astype(np.uint8)
        color_max = np.clip(color_medio + tolerancia, 0, 255).astype(np.uint8)
        
        print(f"    🎨 Color medio: {color_medio}, Tolerancia: {tolerancia}")
        
        # 4. Crear máscara por rango de color (en toda la imagen)
        mascara_color = cv2.inRange(img_base, color_min, color_max)
        
        # 5. BÚSQUEDA ADAPTATIVA
        area_original = np.sum(mascara_original)
        print(f"    📏 Área original: {area_original} píxeles")
        
        # PASO 1: Búsqueda CERCANA
        print(f"    🔍 PASO 1: Búsqueda cercana (radio {zona_cercana}px)...")
        kernel_zona_cercana = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (zona_cercana, zona_cercana))
        zona_busqueda_cercana = cv2.dilate(mascara_original, kernel_zona_cercana)
        mascara_cercana = cv2.bitwise_and(mascara_color, zona_busqueda_cercana)
        
        # Operaciones morfológicas en resultado cercano
        kernel_conectar = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mascara_cercana = cv2.morphologyEx(mascara_cercana, cv2.MORPH_CLOSE, kernel_conectar)
        
        kernel_limpieza = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        mascara_cercana = cv2.morphologyEx(mascara_cercana, cv2.MORPH_OPEN, kernel_limpieza)
        
        # Combinar con original
        mascara_cercana_final = cv2.bitwise_or(mascara_cercana, mascara_original)
        area_cercana = np.sum(mascara_cercana_final)
        mejora_cercana = (area_cercana - area_original) / max(area_original, 1)
        
        print(f"    📊 Área cercana: {area_cercana} píxeles, Mejora: {mejora_cercana:.1%}")
        
        # DECISIÓN: ¿Ampliar búsqueda?
        if mejora_cercana < umbral_mejora:
            # PASO 2: Búsqueda AMPLIA
            print(f"    🔍 PASO 2: Mejora insuficiente ({mejora_cercana:.1%} < {umbral_mejora:.1%})")
            print(f"    🌐 Ampliando búsqueda (radio {zona_amplia}px)...")
            
            kernel_zona_amplia = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (zona_amplia, zona_amplia))
            zona_busqueda_amplia = cv2.dilate(mascara_original, kernel_zona_amplia)
            mascara_amplia = cv2.bitwise_and(mascara_color, zona_busqueda_amplia)
            
            # Operaciones morfológicas en resultado amplio
            mascara_amplia = cv2.morphologyEx(mascara_amplia, cv2.MORPH_CLOSE, kernel_conectar)
            mascara_amplia = cv2.morphologyEx(mascara_amplia, cv2.MORPH_OPEN, kernel_limpieza)
            
            # Combinar con original
            mascara_final = cv2.bitwise_or(mascara_amplia, mascara_original)
            area_final = np.sum(mascara_final)
            mejora_final = (area_final - area_original) / max(area_original, 1)
            
            print(f"    📊 Área amplia: {area_final} píxeles, Mejora final: {mejora_final:.1%}")
            print(f"    ✅ Usando búsqueda AMPLIA")
            
        else:
            # Usar resultado de búsqueda cercana
            mascara_final = mascara_cercana_final
            print(f"    ✅ Usando búsqueda CERCANA (suficiente mejora: {mejora_cercana:.1%})")
        
        return mascara_final
        
    except Exception as e:
        print(f"❌ Error al mejorar máscara de cable: {str(e)}")
        return mascara_original

def visualizar_mejoras_cables(img_base, masks_cables_orig, masks_cables_mejoradas, output_dir):
    """
    Visualiza las máscaras de cables antes y después de la mejora con colores únicos por cable.
    
    Args:
        img_base (np.ndarray): Imagen base
        masks_cables_orig (list): Máscaras originales de YOLO
        masks_cables_mejoradas (list): Máscaras mejoradas
        output_dir (str): Directorio de salida
    """
    try:
        # Generar colores únicos para cada cable
        def generar_colores_cables(num_cables):
            """Genera colores distintos para cada cable"""
            import colorsys
            colores_bgr = []
            for i in range(num_cables):
                # Usar HSV para generar colores bien distribuidos
                hue = i / max(num_cables, 1)  # Distribuir en el espectro
                saturation = 0.8 + (i % 3) * 0.1  # Variar saturación: 0.8, 0.9, 1.0
                value = 0.9 + (i % 2) * 0.1       # Variar brillo: 0.9, 1.0
                
                # Convertir HSV a RGB
                rgb = colorsys.hsv_to_rgb(hue, saturation, value)
                # Convertir a BGR (OpenCV) y escalar a 0-255
                bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
                colores_bgr.append(bgr)
            
            return colores_bgr
        
        num_cables = len(masks_cables_orig)
        colores_cables = generar_colores_cables(num_cables)
        
        # Crear imagen con comparación lado a lado
        h, w = img_base.shape[:2]
        
        # Calcular espacio extra para la leyenda
        espacio_leyenda = max(200, 30 * ((num_cables + 1) // 2))  # Al menos 200px
        img_comparacion = np.zeros((h + espacio_leyenda, w * 2, 3), dtype=np.uint8)
        
        # Lado izquierdo: máscaras originales
        img_izq = img_base.copy()
        for idx, mask in enumerate(masks_cables_orig):
            # Obtener color único para este cable
            color_cable = colores_cables[idx]
            
            mask_bin = (mask.cpu().numpy() > 0.5).astype(np.uint8)
            mask_resized = cv2.resize(mask_bin, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # Dibujar contornos con color único
            contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img_izq, contours, -1, color_cable, 3)  # Grosor 3 para mejor visibilidad
            
            if contours:
                x, y, _, _ = cv2.boundingRect(contours[0])
                # Etiqueta con el mismo color
                cv2.putText(img_izq, f"ORIG C{idx}", (x, y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_cable, 2)
        
        # Lado derecho: máscaras mejoradas
        img_der = img_base.copy()
        for idx, mask_mejorada in enumerate(masks_cables_mejoradas):
            # Usar el mismo color que en el lado izquierdo
            color_cable = colores_cables[idx]
            
            # Dibujar contornos mejorados con el mismo color único
            contours, _ = cv2.findContours(mask_mejorada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img_der, contours, -1, color_cable, 3)  # Grosor 3
            
            if contours:
                x, y, _, _ = cv2.boundingRect(contours[0])
                # Etiqueta con el mismo color
                cv2.putText(img_der, f"MEJ C{idx}", (x, y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_cable, 2)
        
        # Combinar ambas imágenes en la parte superior
        img_comparacion[:h, :w] = img_izq
        img_comparacion[:h, w:] = img_der
        
        # Añadir títulos principales
        cv2.putText(img_comparacion, "MASCARAS ORIGINALES", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
        cv2.putText(img_comparacion, "MASCARAS MEJORADAS", (w + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
        
        # AÑADIR LEYENDA DE COLORES
        y_leyenda_inicio = h + 20
        cv2.putText(img_comparacion, "LEYENDA DE CABLES:", (10, y_leyenda_inicio), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Crear leyenda en dos columnas
        for idx, color_cable in enumerate(colores_cables):
            col = idx % 2  # Columna (0 o 1)
            fila = idx // 2  # Fila
            
            x_leyenda = 20 + col * (w - 50)  # Posición X (columna izquierda o derecha)
            y_leyenda = y_leyenda_inicio + 40 + fila * 35  # Posición Y
            
            # Dibujar cuadrado de color
            cv2.rectangle(img_comparacion, 
                         (x_leyenda, y_leyenda - 15), 
                         (x_leyenda + 25, y_leyenda + 5), 
                         color_cable, -1)
            
            # Borde negro para el cuadrado
            cv2.rectangle(img_comparacion, 
                         (x_leyenda, y_leyenda - 15), 
                         (x_leyenda + 25, y_leyenda + 5), 
                         (0, 0, 0), 1)
            
            # Texto de la leyenda
            cv2.putText(img_comparacion, 
                       f"Cable {idx}", 
                       (x_leyenda + 35, y_leyenda), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Guardar imagen
        output_path = os.path.join(output_dir, "comparacion_mascaras_cables.jpg")
        cv2.imwrite(output_path, img_comparacion)
        print(f"🖼️ Comparación de máscaras guardada en: {output_path}")
        print(f"🎨 Generados {num_cables} colores únicos para cables")
        
        # Estadísticas con colores en terminal (si es compatible)
        print(f"\n📊 ESTADÍSTICAS POR CABLE:")
        for idx, (orig, mejorada) in enumerate(zip(masks_cables_orig, masks_cables_mejoradas)):
            try:
                # Intentar calcular área original
                if hasattr(orig, 'cpu'):
                    # Es tensor de PyTorch
                    orig_bin = (orig.cpu().numpy() > 0.5).astype(np.uint8)
                else:
                    # Ya es numpy array
                    orig_bin = (orig > 0.5).astype(np.uint8)
                
                orig_area = np.sum(orig_bin)
                mejorada_area = np.sum(mejorada)
                mejora_pct = ((mejorada_area - orig_area) / max(orig_area, 1)) * 100
                
                # Mostrar estadística con identificador de color
                color_hex = f"#{colores_cables[idx][2]:02x}{colores_cables[idx][1]:02x}{colores_cables[idx][0]:02x}"
                print(f"🎨 Cable {idx} (Color: {color_hex}): Original={orig_area}px, Mejorada={mejorada_area}px, Mejora={mejora_pct:.1f}%")
                
            except Exception as e:
                print(f"⚠️  Error calculando estadísticas para cable {idx}: {str(e)}")
        
    except Exception as e:
        print(f"❌ Error al visualizar mejoras de cables: {str(e)}")
def encontrar_cable_mas_similar(cables_candidatos, punta_color, img_base, masks_cables_mejoradas, umbral_color=60):
    """
    De una lista de cables candidatos, encuentra el más similar en color y con mayor intersección.
    
    Args:
        cables_candidatos: Lista de (idx_cable, area_interseccion)
        punta_color: Color BGR promedio de la punta
        img_base: Imagen base
        masks_cables_mejoradas: Lista de máscaras mejoradas de cables
        umbral_color: Umbral mínimo de similitud de color (0-100, default: 60)
    
    Returns:
        idx del mejor cable o None si ninguno es suficientemente similar
    """
    if not cables_candidatos:
        return None
    
    print(f"    🎨 Evaluando {len(cables_candidatos)} cables candidatos...")
    print(f"    🎯 Color punta de referencia: {punta_color}")
    print(f"    📏 Umbral de similitud de color: {umbral_color}%")
    
    mejor_cable = None
    mejor_score = -1
    
    for idx_cable, area_interseccion in cables_candidatos:
        # Extraer color promedio del cable
        mask_cable = masks_cables_mejoradas[idx_cable]
        puntos_cable = np.where(mask_cable > 0)
        
        if len(puntos_cable[0]) == 0:
            print(f"      Cable {idx_cable}: Sin píxeles válidos - RECHAZADO")
            continue
            
        colores_cable = img_base[puntos_cable]
        color_cable = np.mean(colores_cable, axis=0).astype(np.uint8)
        
        # Calcular similitud de color (inversa de la distancia euclidiana)
        distancia_color = np.sqrt(np.sum((punta_color - color_cable) ** 2))
        similitud_color = max(0, 100 - distancia_color)  # Entre 0-100
        
        print(f"      Cable {idx_cable}: Color={color_cable}, Distancia={distancia_color:.1f}, Similitud={similitud_color:.1f}%")
        
        # VERIFICACIÓN DE COLOR: Rechazar si es muy diferente
        if similitud_color < umbral_color:
            print(f"      Cable {idx_cable}: Similitud {similitud_color:.1f}% < {umbral_color}% - RECHAZADO por color")
            continue
        
        # Score combinado: similitud de color (70%) + área de intersección (30%)
        area_normalizada = min(100, area_interseccion / 10)  # Normalizar área
        score = similitud_color * 0.7 + area_normalizada * 0.3
        
        print(f"      Cable {idx_cable}: Area={area_interseccion}, Score={score:.1f} - CANDIDATO VÁLIDO")
        
        if score > mejor_score:
            mejor_score = score
            mejor_cable = idx_cable
    
    if mejor_cable is not None:
        print(f"    ✅ Mejor cable: {mejor_cable} (Score: {mejor_score:.1f})")
    else:
        print(f"    ❌ Ningún cable cumple el umbral de color de {umbral_color}%")
    
    return mejor_cable

def buscar_conexiones_expandidas(masks_tips, boxes_tips, masks_cables_mejoradas, img_base, expansion_factor=0.3, umbral_color=60):
    """
    PASO 3: Busca conexiones expandiendo bounding boxes de puntas cuando los métodos anteriores fallan.
    
    Args:
        masks_tips: Lista de máscaras de puntas
        boxes_tips: Lista de bounding boxes de puntas  
        masks_cables_mejoradas: Lista de máscaras mejoradas de cables
        img_base: Imagen base
        expansion_factor: Factor de expansión del bounding box (0.3 = 30%)
        umbral_color: Umbral mínimo de similitud de color (0-100, default: 60)
    
    Returns:
        Lista de coincidencias (idx_punta, coordenadas)
    """
    print(f"  🔍 PASO 3: Búsqueda EXPANDIDA desde puntas (expansión {expansion_factor:.0%}, umbral color {umbral_color}%)...")
    
    coincidencias_expandidas = []
    
    for jdx, (mask_tip, box_tip) in enumerate(zip(masks_tips, boxes_tips)):
        # 1. Expandir bounding box de la punta
        x1, y1, x2, y2 = box_tip.cpu().numpy().astype(int)
        
        # Calcular expansión
        w, h = x2 - x1, y2 - y1
        expansion_w = int(w * expansion_factor)
        expansion_h = int(h * expansion_factor)
        
        # Nuevo bounding box expandido
        x1_exp = max(0, x1 - expansion_w)
        y1_exp = max(0, y1 - expansion_h)
        x2_exp = min(img_base.shape[1], x2 + expansion_w)
        y2_exp = min(img_base.shape[0], y2 + expansion_h)
        
        print(f"    📍 Punta {jdx}: Original=({x1},{y1},{x2},{y2}) → Expandida=({x1_exp},{y1_exp},{x2_exp},{y2_exp})")
        
        # 2. Crear máscara de área expandida
        area_expandida = np.zeros((img_base.shape[0], img_base.shape[1]), dtype=np.uint8)
        area_expandida[y1_exp:y2_exp, x1_exp:x2_exp] = 255
        
        # 3. Buscar cables que intersecten con área expandida
        cables_candidatos = []
        
        for idx_cable, mask_cable in enumerate(masks_cables_mejoradas):
            interseccion = cv2.bitwise_and(mask_cable, area_expandida)
            area_interseccion = np.sum(interseccion)
            
            if area_interseccion > 0:
                cables_candidatos.append((idx_cable, area_interseccion))
                print(f"      🔗 Cable {idx_cable} intersecta con {area_interseccion} píxeles")
        
        print(f"    📊 Encontrados {len(cables_candidatos)} cables candidatos para punta {jdx}")
        
        # 4. Evaluar candidatos por color y área (INCLUSO SI ES SOLO 1)
        if len(cables_candidatos) > 0:
            # Extraer color de la punta para comparación
            mask_tip_bin = mask_tip.cpu().numpy().astype(np.uint8)
            mask_tip_resized = cv2.resize(mask_tip_bin, (img_base.shape[1], img_base.shape[0]), 
                                        interpolation=cv2.INTER_NEAREST)
            puntos_punta = np.where(mask_tip_resized > 0)
            
            if len(puntos_punta[0]) > 0:
                colores_punta = img_base[puntos_punta]
                color_punta = np.mean(colores_punta, axis=0).astype(np.uint8)
                print(f"    🎨 Color punta {jdx}: {color_punta}")
                
                # Encontrar mejor cable (con validación de color)
                mejor_cable = encontrar_cable_mas_similar(
                    cables_candidatos, color_punta, img_base, masks_cables_mejoradas, umbral_color
                )
                
                if mejor_cable is not None:
                    # Calcular coordenadas del centro de la punta
                    x = (x1 + x2) // 2
                    y = (y1 + y2) // 2
                    coincidencias_expandidas.append((jdx, (x, y)))
                    print(f"    ✅ [EXPANDIDA] Punta {jdx} conectada a cable {mejor_cable} en ({x}, {y})")
                else:
                    print(f"    ❌ Punta {jdx}: Cables encontrados pero ninguno es compatible por color")
            else:
                print(f"    ⚠️ Punta {jdx}: No se pudo extraer color - sin píxeles válidos")
        else:
            print(f"    ❌ Punta {jdx}: No intersecta con ningún cable")
    
    print(f"  📊 Resultado PASO 3: {len(coincidencias_expandidas)} puntas encontradas con búsqueda expandida")
    return coincidencias_expandidas


def procesar_cables_y_puntas(img_cables_path, img_puntas_path):
    try:
        # Verificar dimensiones de cables_1024.jpg
        cables_img = cv2.imread(img_cables_path)
        if cables_img is None:
            raise ValueError(f"No se pudo cargar la imagen de cables en {img_cables_path}")
        input_height, input_width = cables_img.shape[:2]
        
        # Factores de escala de 1024x1024 a 3000x3000
        ESCALA_X = 3000 / input_width
        ESCALA_Y = 3000 / input_height
        
        # Detectar cables
        results_cables = modelo_cables.predict(img_cables_path, conf=0.6, save=False)
        names_cables = modelo_cables.names
        masks_cables = []
        for mask, cls_idx in zip(
            results_cables[0].masks.data if results_cables[0].masks is not None else [],
            results_cables[0].boxes.cls if results_cables[0].boxes is not None else []
        ):
            if names_cables[int(cls_idx)] == "wire":
                masks_cables.append(mask)
        print(f"🧵 Detectados {len(masks_cables)} cables")
        
        # Detectar puntas
        results_puntas = modelo_puntas.predict(img_puntas_path, conf=0.3, save=False)
        names_puntas = modelo_puntas.names
        masks_tips = []
        boxes_tips = []
        for mask, cls_idx, box in zip(
            results_puntas[0].masks.data if results_puntas[0].masks is not None else [],
            results_puntas[0].boxes.cls if results_puntas[0].boxes is not None else [],
            results_puntas[0].boxes.xyxy if results_puntas[0].boxes is not None else []
        ):
            if names_puntas[int(cls_idx)] == "tip":
                masks_tips.append(mask)
                boxes_tips.append(box)
        print(f"📍 Detectadas {len(masks_tips)} puntas")
        
        # Visualizar cables y puntas para debugging (detecciones originales)
        visualizar_cables_y_puntas(cables_img, masks_cables, masks_tips, boxes_tips, base_output_dir)
        
        # ESTRATEGIA TRIPLE con debugging detallado
        cables_data = []
        masks_cables_finales = []  # Para visualización (mezcla de originales y mejoradas)
        
        # 🎛️ PARÁMETROS AJUSTABLES
        FACTOR_TOLERANCIA = 0.5   # Factor de tolerancia de color
        TOLERANCIA_BASE = 15       # Tolerancia mínima
        TOLERANCIA_MAX = 20        # Tolerancia máxima
        ZONA_CERCANA = 10          # Radio búsqueda inicial (píxeles)
        ZONA_AMPLIA = 30           # Radio búsqueda ampliada (píxeles) 
        UMBRAL_MEJORA = 0.2        # Umbral para ampliar búsqueda (30%)
        EXPANSION_PUNTAS = 0.2   # Factor de expansión de bounding box de puntas (30%)
        UMBRAL_COLOR = 30
        
        print(f"\n🔧 INICIANDO PROCESAMIENTO TRIPLE ADAPTATIVO DE {len(masks_cables)} CABLES")
        print(f"📊 Parámetros color: Factor={FACTOR_TOLERANCIA}, Base={TOLERANCIA_BASE}, Max={TOLERANCIA_MAX}")
        print(f"📊 Parámetros zona: Cercana={ZONA_CERCANA}px, Amplia={ZONA_AMPLIA}px, Umbral={UMBRAL_MEJORA:.1%}")
        print(f"📊 Parámetros expansión: Puntas={EXPANSION_PUNTAS:.1%}, UmbralColor={UMBRAL_COLOR}%")

        print("=" * 80)
        
        # Contadores para estadísticas
        cables_exitosos_original = 0
        cables_mejorados_cercano = 0
        cables_mejorados_amplio = 0
        cables_expandidos = 0
        cables_fallidos = 0
        
        # Almacenar máscaras mejoradas para PASO 3
        masks_cables_mejoradas_todas = []
        
        for idx, mask_cable in enumerate(masks_cables):
            print(f"\n🔍 CABLE {idx} - Iniciando análisis TRIPLE...")
            
            # Convertir máscara original
            mask_cable_bin = (mask_cable.cpu().numpy() > 0.5).astype(np.uint8)
            mask_resized = cv2.resize(mask_cable_bin, (cables_img.shape[1], cables_img.shape[0]), 
                                    interpolation=cv2.INTER_NEAREST)
            
            # PASO 1: Intentar con máscara original
            print(f"  🎯 PASO 1: Probando máscara ORIGINAL...")
            coincidencias_original = []
            for jdx, mask_tip in enumerate(masks_tips):
                mask_tip_bin = mask_tip.cpu().numpy().astype(np.uint8)
                mask_tip_resized = cv2.resize(mask_tip_bin, (cables_img.shape[1], cables_img.shape[0]), 
                                            interpolation=cv2.INTER_NEAREST)
                
                interseccion = cv2.bitwise_and(mask_resized, mask_tip_resized)
                if np.count_nonzero(interseccion) > 0:
                    box = boxes_tips[jdx].cpu().numpy().astype(int)
                    x = (box[0] + box[2]) // 2
                    y = (box[1] + box[3]) // 2
                    coincidencias_original.append((jdx, (x, y)))
                    print(f"    ✅ Intersección con punta {jdx} en ({x}, {y})")
            
            print(f"  📊 Resultado PASO 1: {len(coincidencias_original)} puntas encontradas")
            
            # Generar máscara mejorada (siempre, para uso potencial en PASO 3)
            mask_cable_mejorada = mejorar_mascara_cable(
                cables_img, mask_resized, 
                factor_tolerancia=FACTOR_TOLERANCIA,
                tolerancia_base=TOLERANCIA_BASE, 
                tolerancia_max=TOLERANCIA_MAX,
                zona_cercana=ZONA_CERCANA,
                zona_amplia=ZONA_AMPLIA,
                umbral_mejora=UMBRAL_MEJORA
            )
            masks_cables_mejoradas_todas.append(mask_cable_mejorada)
            
            # DECISIÓN: ¿Usar original, mejorar, o expandir?
            if len(coincidencias_original) >= 2:
                # 🎯 ÉXITO CON ORIGINAL
                print(f"  🎉 ÉXITO! Cable {idx} resuelto con máscara ORIGINAL")
                print(f"  ➡️  Usando {len(coincidencias_original)} puntas detectadas")
                coincidencias = coincidencias_original
                masks_cables_finales.append(mask_resized)
                cables_exitosos_original += 1
                metodo_usado = "ORIGINAL"
                
            else:
                # 🔧 PASO 2: Probar con máscara mejorada
                print(f"  ⚠️  Cable {idx} insuficiente con original ({len(coincidencias_original)} puntas)")
                print(f"  🔧 PASO 2: Probando máscara MEJORADA...")
                
                coincidencias_mejorada = []
                for jdx, mask_tip in enumerate(masks_tips):
                    mask_tip_bin = mask_tip.cpu().numpy().astype(np.uint8)
                    mask_tip_resized = cv2.resize(mask_tip_bin, (cables_img.shape[1], cables_img.shape[0]), 
                                                interpolation=cv2.INTER_NEAREST)
                    
                    interseccion = cv2.bitwise_and(mask_cable_mejorada, mask_tip_resized)
                    if np.count_nonzero(interseccion) > 0:
                        box = boxes_tips[jdx].cpu().numpy().astype(int)
                        x = (box[0] + box[2]) // 2
                        y = (box[1] + box[3]) // 2
                        coincidencias_mejorada.append((jdx, (x, y)))
                        print(f"    ✅ [MEJORADA] Intersección con punta {jdx} en ({x}, {y})")
                
                print(f"  📊 Resultado PASO 2: {len(coincidencias_mejorada)} puntas encontradas")
                
                if len(coincidencias_mejorada) >= 2:
                    # 🔧 ÉXITO CON MEJORADA
                    print(f"  🎉 ÉXITO! Cable {idx} resuelto con máscara MEJORADA")
                    print(f"  ➡️  Mejora exitosa: {len(coincidencias_original)} → {len(coincidencias_mejorada)} puntas")
                    
                    # Determinar si usó búsqueda cercana o amplia
                    area_mejorada = np.sum(mask_cable_mejorada)
                    area_original = np.sum(mask_resized) 
                    mejora_ratio = (area_mejorada - area_original) / max(area_original, 1)
                    
                    if mejora_ratio >= UMBRAL_MEJORA * 2:
                        cables_mejorados_amplio += 1
                        metodo_usado = "MEJORADA_AMPLIA"
                    else:
                        cables_mejorados_cercano += 1  
                        metodo_usado = "MEJORADA_CERCANA"
                    
                    coincidencias = coincidencias_mejorada
                    masks_cables_finales.append(mask_cable_mejorada)
                    
                else:
                    # 🌐 PASO 3: Búsqueda expandida desde puntas (ÚLTIMO RECURSO)
                    print(f"  ⚠️  Cable {idx} insuficiente con mejora ({len(coincidencias_mejorada)} puntas)")
                    print(f"  🌐 PASO 3: Iniciando búsqueda EXPANDIDA (último recurso)...")
                    
                    # Buscar solo para este cable usando expansión de puntas
                    coincidencias_expandida = []
                    masks_cable_actual = [mask_cable_mejorada]  # Solo este cable
                    
                    # Buscar conexiones expandidas
                    coincidencias_temp = buscar_conexiones_expandidas(
                        masks_tips, boxes_tips, masks_cable_actual, cables_img, EXPANSION_PUNTAS, UMBRAL_COLOR
                    )
                    
                    # Filtrar solo las conexiones que corresponden a este cable (índice 0 en masks_cable_actual)
                    for jdx, coords in coincidencias_temp:
                        coincidencias_expandida.append((jdx, coords))
                    
                    print(f"  📊 Resultado PASO 3: {len(coincidencias_expandida)} puntas encontradas con expansión")
                    
                    if len(coincidencias_expandida) >= 2:
                        print(f"  🎉 ÉXITO! Cable {idx} resuelto con búsqueda EXPANDIDA")
                        print(f"  ➡️  Expansión exitosa: {len(coincidencias_mejorada)} → {len(coincidencias_expandida)} puntas")
                        cables_expandidos += 1
                        metodo_usado = "EXPANDIDA"
                        coincidencias = coincidencias_expandida
                    else:
                        print(f"  ❌ FALLO! Cable {idx} no resuelto ni con expansión")
                        cables_fallidos += 1
                        metodo_usado = "FALLIDO"
                        coincidencias = coincidencias_expandida
                    
                    masks_cables_finales.append(mask_cable_mejorada)
            
            # Procesar las coincidencias encontradas
            if len(coincidencias) >= 2:
                if len(coincidencias) > 2:
                    print(f"  ⚠️  Múltiples puntas ({len(coincidencias)}) - seleccionando las 2 más distantes")
                    puntos = [coord for _, coord in coincidencias]
                    distancias = []
                    for i in range(len(puntos)):
                        for j in range(i + 1, len(puntos)):
                            dist = np.sqrt((puntos[i][0] - puntos[j][0])**2 + 
                                         (puntos[i][1] - puntos[j][1])**2)
                            distancias.append((dist, i, j))
                    distancias.sort(reverse=True)
                    _, i, j = distancias[0]
                    idx1, idx2 = coincidencias[i][0], coincidencias[j][0]
                    print(f"  📏 Puntas más distantes: {idx1} y {idx2} (distancia: {distancias[0][0]:.1f}px)")
                else:
                    idx1, idx2 = coincidencias[0][0], coincidencias[1][0]
                    print(f"  📍 Usando puntas: {idx1} y {idx2}")
                
                box1 = boxes_tips[idx1].cpu().numpy().astype(int)
                box2 = boxes_tips[idx2].cpu().numpy().astype(int)
                x1 = (box1[0] + box1[2]) // 2
                y1 = (box1[1] + box1[3]) // 2
                x2 = (box2[0] + box2[2]) // 2
                y2 = (box2[1] + box2[3]) // 2
                
                x1_scaled = int(x1 * ESCALA_X)
                y1_scaled = int(y1 * ESCALA_Y)
                x2_scaled = int(x2 * ESCALA_X)
                y2_scaled = int(y2 * ESCALA_Y)
                
                cables_data.append({
                    "id": idx,
                    "extremo1": (x1_scaled, y1_scaled),
                    "extremo2": (x2_scaled, y2_scaled)
                })
                
                print(f"  ✅ Cable {idx} COMPLETADO [{metodo_usado}]")
                print(f"    📍 Extremo 1: ({x1_scaled}, {y1_scaled})")
                print(f"    📍 Extremo 2: ({x2_scaled}, {y2_scaled})")
            else:
                print(f"  ❌ Cable {idx} RECHAZADO - Solo {len(coincidencias)} puntas")
        
        # ESTADÍSTICAS FINALES AMPLIADAS
        print("\n" + "=" * 80)
        print("📊 RESUMEN FINAL DEL PROCESAMIENTO TRIPLE ADAPTATIVO")
        print("=" * 80)
        print(f"🎯 Cables exitosos con máscara ORIGINAL: {cables_exitosos_original}")
        print(f"🔧 Cables mejorados con búsqueda CERCANA: {cables_mejorados_cercano}")
        print(f"🌐 Cables mejorados con búsqueda AMPLIA: {cables_mejorados_amplio}")
        print(f"🔗 Cables salvados con búsqueda EXPANDIDA: {cables_expandidos}")
        print(f"❌ Cables FALLIDOS: {cables_fallidos}")
        print(f"✅ Total cables EXITOSOS: {len(cables_data)}")
        print(f"📊 Tasa de éxito: {(len(cables_data)/len(masks_cables)*100):.1f}%")
        
        total_mejorados = cables_mejorados_cercano + cables_mejorados_amplio + cables_expandidos
        if total_mejorados > 0:
            print(f"🔧 Efectividad de mejoras: {(total_mejorados/(total_mejorados+cables_fallidos)*100):.1f}%")
            print(f"📊 Distribución: {cables_mejorados_cercano} cercanas, {cables_mejorados_amplio} amplias, {cables_expandidos} expandidas")
        
        # Visualizar comparación (Original vs Triple Adaptativo)
        print(f"\n🖼️  Generando visualización: ORIGINAL vs TRIPLE ADAPTATIVO...")
        visualizar_mejoras_cables(cables_img, masks_cables, masks_cables_finales, base_output_dir)
        
        cables_df = pd.DataFrame([
            {
                "id": c["id"],
                "extremo1_x": c["extremo1"][0],
                "extremo1_y": c["extremo1"][1],
                "extremo2_x": c["extremo2"][0],
                "extremo2_y": c["extremo2"][1],
                "tipo": "cable"
            }
            for c in cables_data
        ])
        
        print(f"\n📋 DataFrame final de cables ({len(cables_data)} exitosos):")
        print(cables_df)
        return cables_df
        
    except Exception as e:
        print(f"❌ Error al procesar cables: {str(e)}")
        return pd.DataFrame(columns=["id", "extremo1_x", "extremo1_y", "extremo2_x", "extremo2_y", "tipo"])
# --- Funciones de visualización ---
def generar_imagen_componentes(base_img, resistencias_df, leds_df, botones_df, cables_df, cristales_df, cap_cers_df, usonic_detections, dips_df=None):
    """
    Genera una imagen con los componentes dibujados a partir de los DataFrames y detecciones proporcionados.
    
    Args:
        base_img (np.ndarray): Imagen base.
        resistencias_df (pd.DataFrame): DataFrame con resistencias.
        leds_df (pd.DataFrame): DataFrame con LEDs.
        botones_df (pd.DataFrame): DataFrame con botones.
        cables_df (pd.DataFrame): DataFrame con cables.
        cristales_df (pd.DataFrame): DataFrame con cristales.
        cap_cers_df (pd.DataFrame): DataFrame con capacitores cerámicos.
        usonic_detections (list): Lista de diccionarios con detecciones de ultrasonidos.
        dips_df (pd.DataFrame, optional): DataFrame con pines de DIPs.
    
    Returns:
        None: Guarda la imagen generada en el directorio especificado.
    """
    try:
        # Convertir imagen base a RGB
        imagen_final_df = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB).copy()
        
        # Unificar DataFrames de componentes con extremos
        todos_df = pd.concat([resistencias_df, leds_df, botones_df, cables_df, cristales_df, cap_cers_df], ignore_index=True)
        
        # Dibujar componentes con extremos
        for idx, row in todos_df.iterrows():
            if pd.notna(row["extremo1_x"]) and pd.notna(row["extremo2_x"]):
                p1 = (int(row["extremo1_x"]), int(row["extremo1_y"]))
                p2 = (int(row["extremo2_x"]), int(row["extremo2_y"]))
                
                color = {
                    "resistencia": (255, 0, 0),    # Rojo
                    "led": (0, 255, 0),           # Verde
                    "boton": (0, 200, 255),       # Naranja
                    "cable": (200, 200, 0),       # Amarillo
                    "cristal": (0, 255, 255),      # Cian
                    "capacitor": (255, 0, 255)    # Magenta
                }.get(row["tipo"], (255, 255, 255))  # Blanco por defecto
                
                cv2.circle(imagen_final_df, p1, 8, color, -1)
                cv2.circle(imagen_final_df, p2, 8, color, -1)
                cv2.line(imagen_final_df, p1, p2, color, 2)
                cv2.putText(imagen_final_df, f"{row['tipo'][:3].upper()}{row['id']}", p1, 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Dibujar pines de DIPs si existen
        if dips_df is not None and not dips_df.empty:
            for idx, row in dips_df.iterrows():
                if pd.notna(row["x"]) and pd.notna(row["y"]):
                    p = (int(row["x"]), int(row["y"]))
                    color = (255, 0, 255)  # Magenta para DIPs
                    cv2.circle(imagen_final_df, p, 8, color, -1)
                    cv2.putText(imagen_final_df, f"{row['pin_id_u']}", p, 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Dibujar detecciones de ultrasonidos
        if usonic_detections and len(usonic_detections) > 0:
            for usonic in usonic_detections:
                # Dibujar bounding box original
                x1, y1, x2, y2 = map(int, usonic["box"])
                color_usonic = (64, 224, 208)  # Turquesa para ultrasonidos
                cv2.rectangle(imagen_final_df, (x1, y1), (x2, y2), color_usonic, 2)
                cv2.putText(imagen_final_df, f"USO{usonic['id']}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_usonic, 2)
                
                # Dibujar pines numerados
                for pin in usonic["pins"]:
                    p = (int(pin["x"]), int(pin["y"]))
                    cv2.circle(imagen_final_df, p, 8, color_usonic, -1)
                    cv2.putText(imagen_final_df, str(pin["number"]), p,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_usonic, 2)
        
        # Guardar imagen
        output_todos_path = os.path.join(base_output_dir, "todos_componentes_desde_dataframe_y_cables.png")
        cv2.imwrite(output_todos_path, cv2.cvtColor(imagen_final_df, cv2.COLOR_RGB2BGR))
        print(f"\n🖼️ Imagen con todos los componentes desde el DataFrame guardada en:\n{output_todos_path}")
        
    except Exception as e:
        print(f"❌ Error al generar imagen de componentes: {str(e)}")

def unificar_dataframes(resistencias_df, leds_df, botones_df, cables_df, usonic_detections, cristales_df, cap_cers_df, dips_df=None):
    """
    Unifica los DataFrames de componentes en un formato estandarizado con IDs únicos y pines.
    Conserva los tipos específicos de DIPs (por ejemplo, 7408, 7432). Guarda en processing_state.
    
    Args:
        resistencias_df (pd.DataFrame): DataFrame de resistencias.
        leds_df (pd.DataFrame): DataFrame de LEDs.
        botones_df (pd.DataFrame): DataFrame de botones.
        cables_df (pd.DataFrame): DataFrame de cables.
        usonic_detections (list): Lista de diccionarios con detecciones de ultrasonidos.
        cristales_df (pd.DataFrame): DataFrame de cristales.
        cap_cers_df (pd.DataFrame): DataFrame de capacitores cerámicos.
        dips_df (pd.DataFrame, optional): DataFrame de pines de DIPs. Defaults to None.
    
    Returns:
        pd.DataFrame: DataFrame unificado con todos los componentes.
    """
    global processing_state
    
    try:
        # Crear copias para no modificar los originales
        resistencias_aux = resistencias_df.copy() if resistencias_df is not None else pd.DataFrame()
        leds_aux = leds_df.copy() if leds_df is not None else pd.DataFrame()
        botones_aux = botones_df.copy() if botones_df is not None else pd.DataFrame()
        cables_aux = cables_df.copy() if cables_df is not None else pd.DataFrame()
        cristales_aux = cristales_df.copy() if cristales_df is not None else pd.DataFrame()
        cap_cers_aux = cap_cers_df.copy() if cap_cers_df is not None else pd.DataFrame()
        dips_aux = dips_df.copy() if dips_df is not None and not dips_df.empty else pd.DataFrame()
        
        # Aplicar offsets a los IDs para evitar duplicados
        if not resistencias_aux.empty:
            resistencias_aux["id"] = resistencias_aux["id"] + 1000
        if not leds_aux.empty:
            leds_aux["id"] = leds_aux["id"] + 2000
        if not botones_aux.empty:
            botones_aux["id"] = botones_aux["id"] + 3000
        if not cables_aux.empty:
            cables_aux["id"] = cables_aux["id"] + 4000
        # Ultrasonidos tendrán IDs 5000-5999
        usonic_ids = {}
        if usonic_detections and len(usonic_detections) > 0:
            for idx, usonic in enumerate(usonic_detections):
                usonic_ids[usonic['id']] = 5000 + idx
        if not cristales_aux.empty:
            cristales_aux["id"] = cristales_aux["id"] + 6000
        if not cap_cers_aux.empty:
            cap_cers_aux["id"] = cap_cers_aux["id"] + 7000
        # DIPs mantienen sus IDs (0-999)
        
        # Unir los DataFrames de componentes lineales
        todos_aux = pd.concat([resistencias_aux, leds_aux, botones_aux, cables_aux, cristales_aux, cap_cers_aux], ignore_index=True)
        
        # Convertir componentes lineales en puntos (pin_num 1 y 2)
        extra_rows = []
        for idx, row in todos_aux.iterrows():
            if pd.notna(row['extremo1_x']) and pd.notna(row['extremo2_x']):
                extremos = [
                    (row['extremo1_x'], row['extremo1_y'], '1'),
                    (row['extremo2_x'], row['extremo2_y'], '2')
                ]
                for coord_x, coord_y, pin_num in extremos:
                    extra_rows.append({
                        'componente_id': row['id'],
                        'tipo_componente': row['tipo'],
                        'pin_num': pin_num,
                        'pin_nombre': 'pin',
                        'x': float(coord_x),
                        'y': float(coord_y)
                    })
        
        # Convertir ultrasonidos en filas (pins con nombres fijos)
        usonic_rows = []
        if usonic_detections and len(usonic_detections) > 0:
            for usonic in usonic_detections:
                usonic_id = usonic_ids[usonic['id']]
                for pin in usonic['pins']:
                    pin_num = str(pin['number'])
                    pin_nombre = {
                        '1': 'VCC',
                        '2': 'Trig',
                        '3': 'Echo',
                        '4': 'GND'
                    }.get(pin_num, f'Pin{pin_num}')  # Fallback si el número no es 1-4
                    usonic_rows.append({
                        'componente_id': usonic_id,
                        'tipo_componente': 'usonic',
                        'pin_num': pin_num,
                        'pin_nombre': pin_nombre,
                        'x': float(pin['x']),
                        'y': float(pin['y'])
                    })
        
        # Crear DataFrame de componentes lineales y ultrasonidos
        otros_df = pd.DataFrame(extra_rows)
        usonic_df = pd.DataFrame(usonic_rows)
        otros_df = pd.concat([otros_df, usonic_df], ignore_index=True)
        
        # Preparar dips_df para unificación
        if not dips_aux.empty:
            dips_aux = dips_aux.rename(columns={
                'dip_id': 'componente_id',
                'pin_id_u': 'pin_num',
                'pin_name': 'pin_nombre'
            })
            # Conservar tipo_componente específico (7408, 7432, etc.)
            dips_aux = dips_aux[['componente_id', 'tipo_componente', 'pin_num', 'pin_nombre', 'x', 'y']]
        
        # Unificar todos los componentes
        unificado_df = pd.concat([otros_df, dips_aux], ignore_index=True)
        
        # Eliminar columnas innecesarias si existen
        columnas_a_eliminar = ['detectado', 'interpolado', 'fila', 'posicion_muesca']
        unificado_df = unificado_df.drop(columns=[col for col in columnas_a_eliminar if col in unificado_df.columns])
        
        # Guardar en processing_state
        processing_state['unificado_df'] = unificado_df
        
        # Imprimir DataFrame unificado
        print("\n📋 DataFrame unificado:")
        print(unificado_df.to_string())
        
        # Imprimir información sobre los rangos de IDs
        print("\n📋 Rangos de IDs asignados:")
        print(f"- DIPs (7408, 7432, etc.): 0-999")
        print(f"- Resistencias: 1000-1999")
        print(f"- LEDs: 2000-2999")
        print(f"- Botones: 3000-3999")
        print(f"- Cables: 4000-4999")
        print(f"- Ultrasonidos: 5000-5999")
        print(f"- Cristales: 6000-6999")
        print(f"- Capacitores cerámicos: 7000-7999")
        
        # Verificar IDs duplicados
        ids_unicos = unificado_df['componente_id'].nunique()
        total_componentes = len(unificado_df['componente_id'].drop_duplicates())
        if ids_unicos == total_componentes:
            print(f"\n✅ ¡Éxito! Todos los IDs de componente son únicos ({ids_unicos} componentes)")
        else:
            print(f"\n⚠️ ¡Advertencia! Hay IDs duplicados. {ids_unicos} IDs únicos vs {total_componentes} componentes")
        
        return unificado_df
    
    except Exception as e:
        print(f"❌ Error al unificar DataFrames: {str(e)}")
        unificado_df = pd.DataFrame(columns=['componente_id', 'tipo_componente', 'pin_num', 'pin_nombre', 'x', 'y'])
        processing_state['unificado_df'] = unificado_df
        return unificado_df
        
import time  # Asegúrate de que este import esté al inicio del script

def generar_conexiones_carriles(unificado_df):
    """
    Genera un DataFrame de conexiones basado en carriles del protoboard y zonas de alimentación,
    usando el DataFrame unificado. Almacena las conexiones en processing_state['conexiones_df'].
    Dibuja los carriles filtrados en la imagen base para debug.
    
    Args:
        unificado_df (pd.DataFrame): DataFrame unificado con componentes y pines.
    
    Returns:
        pd.DataFrame: DataFrame con todas las conexiones (carriles y power).
    """
    global processing_state
    
    try:
        # Cargar modelos
        modelo_zonas = YOLO("modelos/breadboardZones.pt")
        modelo_carriles = YOLO("modelos/carrilesProtoboard.pt")
        
        # Cargar imagen 3000x3000
        img_path = "preprocesamiento_individual/3000x3000.jpg"
        img_3000 = cv2.imread(img_path)
        if img_3000 is None:
            raise ValueError(f"No se pudo cargar la imagen en {img_path}")
        
        # Detectar zonas (power y middle)
        results_zonas = modelo_zonas.predict(img_3000, conf=0.3)
        zonas_data = []
        for box, cls_idx in zip(results_zonas[0].boxes.xyxy, results_zonas[0].boxes.cls):
            x1, y1, x2, y2 = box.cpu().numpy()
            clase = modelo_zonas.names[int(cls_idx)]
            zonas_data.append({
                "clase": clase,
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2)
            })
        
        zonas_df = pd.DataFrame(zonas_data)
        zonas_power = zonas_df[zonas_df["clase"] == "power"]
        zona_middle = zonas_df[zonas_df["clase"] == "middle"]
        
        print(f"\n📊 Zonas detectadas: {len(zonas_power)} power, {len(zona_middle)} middle")
        
        # Función para verificar si un punto está en una zona prohibida
        def punto_en_zona_prohibida(x, y):
            for _, zona in zonas_power.iterrows():
                if zona["x1"] <= x <= zona["x2"] and zona["y1"] <= y <= zona["y2"]:
                    return True
            for _, zona in zona_middle.iterrows():
                if zona["x1"] <= x <= zona["x2"] and zona["y1"] <= y <= zona["y2"]:
                    return True
            return False
        
        # Filtrar puntos fuera de zonas prohibidas
        puntos_filtrados = []
        for idx, row in unificado_df.iterrows():
            x, y = float(row["x"]), float(row["y"])
            if not punto_en_zona_prohibida(x, y):
                puntos_filtrados.append({
                    "id": idx,
                    "componente_id": row["componente_id"],
                    "tipo_componente": row["tipo_componente"],
                    "pin_num": row["pin_num"],
                    "pin_nombre": row["pin_nombre"],
                    "x": x,
                    "y": y
                })
        
        puntos_filtrados_df = pd.DataFrame(puntos_filtrados)
        print(f"\n📊 Puntos fuera de zonas prohibidas: {len(puntos_filtrados_df)} de {len(unificado_df)}")
        
        # Detectar carriles para puntos filtrados
        zonas_ordenadas = zonas_df.sort_values(by="y1").to_dict("records")
        carriles_validos = []
        
        for idx, punto in puntos_filtrados_df.iterrows():
            x, y = int(punto["x"]), int(punto["y"])
            
            # Encontrar cajas superior e inferior
            y1, y2 = None, None
            for j in range(len(zonas_ordenadas) - 1):
                y_sup = zonas_ordenadas[j]["y2"]
                y_inf = zonas_ordenadas[j+1]["y1"]
                if y_sup <= y <= y_inf:
                    y1, y2 = y_sup, y_inf
                    break
            
            if y1 is None or y2 is None:
                y1 = max(0, y - 150)
                y2 = min(img_3000.shape[0], y + 150)
            
            x1 = max(0, x - 70)
            x2 = min(img_3000.shape[1], x + 70)
            
            # Recortar región
            recorte = img_3000[y1:y2, x1:x2]
            if recorte.size == 0:
                print(f"⚠️ Recorte vacío para punto {idx}, se omite.")
                continue
            
            # Inferencia de carriles
            results = modelo_carriles.predict(recorte, conf=0.5)
            for box, conf in zip(results[0].boxes.xyxy, results[0].boxes.conf):
                x1_box, y1_box, x2_box, y2_box = [int(val) for val in box.cpu().numpy()]
                x1_abs = x1 + x1_box
                y1_abs = y1 + y1_box
                x2_abs = x1 + x2_box
                y2_abs = y1 + y2_box
                
                if x1_abs <= x <= x2_abs and y1_abs <= y <= y2_abs:
                    # Ajustar los límites y1_abs y y2_abs al recorte completo
                    y1_abs = y1
                    y2_abs = y2
                    carriles_validos.append({
                        "x1": x1_abs,
                        "y1": y1_abs,
                        "x2": x2_abs,
                        "y2": y2_abs,
                        "punto_id": punto["id"],
                        "componente_id": punto["componente_id"],
                        "tipo_componente": punto["tipo_componente"],
                        "pin_num": punto["pin_num"],
                        "pin_nombre": punto["pin_nombre"],
                        "confianza": float(conf)
                    })
                    print(f"✅ Carril encontrado para punto {idx} ({punto['componente_id']}_{punto['pin_num']})")
                    break
        
        # Filtrar carriles repetidos (superposición > 90%)
        def calcular_superposicion(box1, box2):
            x1_inter = max(box1["x1"], box2["x1"])
            y1_inter = max(box1["y1"], box2["y1"])
            x2_inter = min(box1["x2"], box2["x2"])
            y2_inter = min(box1["y2"], box2["y2"])
            if x1_inter >= x2_inter or y1_inter >= y2_inter:
                return 0
            area_inter = (x2_inter - x1_inter) * (y2_inter - y1_inter)
            area_box1 = (box1["x2"] - box1["x1"]) * (box1["y2"] - box1["y1"])
            area_box2 = (box2["x2"] - box2["x1"]) * (box2["y2"] - box2["y1"])
            return (area_inter / min(area_box1, area_box2)) * 100 if min(area_box1, area_box2) > 0 else 0
        
        carriles_filtrados = []
        for carril in carriles_validos:
            es_repetido = False
            for carril_filtrado in carriles_filtrados:
                if calcular_superposicion(carril, carril_filtrado) > 90:
                    es_repetido = True
                    break
            if not es_repetido:
                carriles_filtrados.append(carril)
        
        print(f"\n📊 Carriles únicos: {len(carriles_filtrados)} de {len(carriles_validos)} iniciales")
        
        # Dibujar carriles filtrados en la imagen base para debug
        img_debug = img_3000.copy()
        color_carril = (144, 238, 144)  # Verde claro para carriles
        for idx, carril in enumerate(carriles_filtrados):
            x1, y1, x2, y2 = carril["x1"], carril["y1"], carril["x2"], carril["y2"]
            cv2.rectangle(img_debug, (x1, y1), (x2, y2), color_carril, 2)
            cv2.putText(img_debug, f"C{idx}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_carril, 2)
        
        # Guardar imagen de debug
        os.makedirs(os.path.join(base_output_dir, "carriles"), exist_ok=True)
        debug_path = os.path.join(base_output_dir, "carriles", f"debug_carriles_{int(time.time())}.jpg")
        cv2.imwrite(debug_path, img_debug)
        print(f"\n🖼️ Imagen de debug con carriles filtrados guardada en: {debug_path}")
        
        # Generar conexiones para carriles
        conexiones = []
        conexion_id = 0
        for i, carril in enumerate(carriles_filtrados):
            puntos_en_carril = puntos_filtrados_df[
                (puntos_filtrados_df["x"] >= carril["x1"]) & 
                (puntos_filtrados_df["x"] <= carril["x2"]) & 
                (puntos_filtrados_df["y"] >= carril["y1"]) & 
                (puntos_filtrados_df["y"] <= carril["y2"])
            ]
            
            if len(puntos_en_carril) >= 2:
                for _, punto in puntos_en_carril.iterrows():
                    conexiones.append({
                        "conexion_id": conexion_id,
                        "carril_id": i,
                        "componente_id": punto["componente_id"],
                        "tipo_componente": punto["tipo_componente"],
                        "pin_num": punto["pin_num"],
                        "pin_nombre": punto["pin_nombre"],
                        "x": punto["x"],
                        "y": punto["y"],
                        "zona_power": None
                    })
                print(f"✅ Conexión #{conexion_id} creada para carril {i} with {len(puntos_en_carril)} puntos")
                conexion_id += 1
        
        conexiones_df = pd.DataFrame(conexiones)
        
        # Procesar puntos en zonas de alimentación (power)
        conexiones_power = []
        for i, zona in zonas_power.iterrows():
            x1, y1, x2, y2 = zona["x1"], zona["y1"], zona["x2"], zona["y2"]
            altura_zona = y2 - y1
            limite_superior = y1 + altura_zona * 0.4
            limite_inferior = y1 + altura_zona * 0.6
            
            puntos_en_power = unificado_df[
                (unificado_df["x"] >= x1) & 
                (unificado_df["x"] <= x2) & 
                (unificado_df["y"] >= y1) & 
                (unificado_df["y"] <= y2)
            ]
            
            if len(puntos_en_power) > 0:
                puntos_fila_superior = puntos_en_power[puntos_en_power["y"] <= limite_superior]
                puntos_fila_inferior = puntos_en_power[puntos_en_power["y"] >= limite_inferior]
                puntos_medio = puntos_en_power[
                    (puntos_en_power["y"] > limite_superior) & 
                    (puntos_en_power["y"] < limite_inferior)
                ]
                
                for _, punto in puntos_medio.iterrows():
                    dist_superior = abs(punto["y"] - y1)
                    dist_inferior = abs(punto["y"] - y2)
                    if dist_superior <= dist_inferior:
                        puntos_fila_superior = pd.concat([puntos_fila_superior, pd.DataFrame([punto])], ignore_index=True)
                    else:
                        puntos_fila_inferior = pd.concat([puntos_fila_inferior, pd.DataFrame([punto])], ignore_index=True)
                
                if len(puntos_fila_superior) >= 2:
                    for _, punto in puntos_fila_superior.iterrows():
                        conexiones_power.append({
                            "conexion_id": conexion_id,
                            "carril_id": -1,
                            "componente_id": punto["componente_id"],
                            "tipo_componente": punto["tipo_componente"],
                            "pin_num": punto["pin_num"],
                            "pin_nombre": punto["pin_nombre"],
                            "x": punto["x"],
                            "y": punto["y"],
                            "zona_power": f"power_{i}_superior"
                        })
                    print(f"✅ Conexión power #{conexion_id} creada para fila superior de zona {i}")
                    conexion_id += 1
                
                if len(puntos_fila_inferior) >= 2:
                    for _, punto in puntos_fila_inferior.iterrows():
                        conexiones_power.append({
                            "conexion_id": conexion_id,
                            "carril_id": -1,
                            "componente_id": punto["componente_id"],
                            "tipo_componente": punto["tipo_componente"],
                            "pin_num": punto["pin_num"],
                            "pin_nombre": punto["pin_nombre"],
                            "x": punto["x"],
                            "y": punto["y"],
                            "zona_power": f"power_{i}_inferior"
                        })
                    print(f"✅ Conexión power #{conexion_id} creada para fila inferior de zona {i}")
                    conexion_id += 1
        
        conexiones_power_df = pd.DataFrame(conexiones_power)
        
        # Unir conexiones de carriles y power
        conexiones_completas_df = pd.concat([conexiones_df, conexiones_power_df], ignore_index=True)
        
        # Guardar en processing_state
        processing_state['conexiones_df'] = conexiones_completas_df
        
        # Imprimir resúmenes
        print("\n📋 Resumen de conexiones:")
        if not conexiones_df.empty:
            conexiones_por_carril = conexiones_df.groupby("carril_id").size().reset_index(name="num_puntos")
            print("\nConexiones por carriles:")
            for _, row in conexiones_por_carril.iterrows():
                print(f"- Carril {int(row['carril_id'])}: {int(row['num_puntos'])} puntos")
        
        if not conexiones_power_df.empty:
            conexiones_por_power = conexiones_power_df.groupby("zona_power").size().reset_index(name="num_puntos")
            print("\nConexiones por zonas power:")
            for _, row in conexiones_por_power.iterrows():
                print(f"- {row['zona_power']}: {int(row['num_puntos'])} puntos")
        
        print(f"\n📊 Resumen final:")
        print(f"- Puntos totales: {len(unificado_df)}")
        print(f"- Puntos fuera de zonas prohibidas: {len(puntos_filtrados_df)}")
        print(f"- Conexiones por carriles: {len(conexiones_df['conexion_id'].unique())}")
        print(f"- Conexiones por power: {len(conexiones_power_df['conexion_id'].unique())}")
        print(f"- Total conexiones: {len(conexiones_completas_df['conexion_id'].unique())}")
        print(f"- Puntos conectados: {len(conexiones_completas_df)}")
        
        print("\n📋 DataFrame de conexiones:")
        print(conexiones_completas_df.to_string())
        
        return conexiones_completas_df
    
    except Exception as e:
        print(f"❌ Error al generar conexiones: {str(e)}")
        conexiones_completas_df = pd.DataFrame(columns=[
            "conexion_id", "carril_id", "componente_id", "tipo_componente",
            "pin_num", "pin_nombre", "x", "y", "zona_power"
        ])
        processing_state['conexiones_df'] = conexiones_completas_df
        return conexiones_completas_df
    
def visualizar_detecciones_componentes(base_img, all_boxes, all_classes, all_confs, class_names, output_dir):
    """
    Dibuja los bounding boxes de todas las detecciones de componentes en la imagen base y guarda el resultado.
    Solo para debugging, no se envía a la app móvil.
    
    Args:
        base_img (np.ndarray): Imagen base (3000x3000).
        all_boxes (list): Lista de bounding boxes [x1, y1, x2, y2].
        all_classes (list): Lista de IDs de clases de las detecciones.
        all_confs (list): Lista de valores de confianza de las detecciones.
        class_names (dict): Diccionario que mapea IDs de clase a nombres (ej. {0: 'res', 1: 'led'}).
        output_dir (str): Directorio donde guardar la imagen resultante.
    """
    try:
        # Crear copia de la imagen base
        img_debug = base_img.copy()
        
        # Definir colores por clase (BGR)
        colores = {
            'res': (0, 0, 255),      # Rojo para resistencias
            'led': (255, 0, 0),      # Azul para LEDs
            'button': (0, 255, 0),   # Verde para botones
            'dip': (255, 255, 0),    # Amarillo para DIPs
        }
        
        # Dibujar cada detección
        for box, cls_id, conf in zip(all_boxes, all_classes, all_confs):
            x1, y1, x2, y2 = box
            # Asegurar que las coordenadas estén dentro de la imagen
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, img_debug.shape[1]), min(y2, img_debug.shape[0])
            
            # Obtener nombre de la clase
            class_name = class_names.get(cls_id, f'cls_{cls_id}')
            # Obtener color (por defecto morado si la clase no está en colores)
            color = colores.get(class_name, (255, 0, 255))
            
            # Dibujar bounding box
            cv2.rectangle(img_debug, (x1, y1), (x2, y2), color, 2)
            
            # Crear etiqueta con nombre y confianza
            label = f'{class_name} {conf:.2f}'
            # Dibujar fondo para la etiqueta (para mejor legibilidad)
            (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_debug, (x1, y1 - label_h - baseline), 
                         (x1 + label_w, y1), color, -1)
            # Dibujar texto
            cv2.putText(img_debug, label, (x1, y1 - baseline), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Guardar la imagen
        output_path = os.path.join(output_dir, 'detecciones_componentes.jpg')
        cv2.imwrite(output_path, img_debug)
        print(f"🖼️ Imagen de debugging guardada en: {output_path}")
        
    except Exception as e:
        print(f"❌ Error al generar imagen de debugging: {str(e)}")

def visualizar_conexiones(conexiones_df):
    """
    Dibuja puntos y líneas en la imagen 3000x3000 para visualizar conexiones basadas en conexiones_df.
    Guarda la imagen con leyenda en preprocesamiento_individual/conexiones_encontradas.png.
    
    Args:
        conexiones_df (pd.DataFrame): DataFrame con conexiones (conexion_id, x, y, etc.).
    
    Returns:
        bool: True si la visualización se generó correctamente, False si hubo un error.
    """
    try:
        # Cargar imagen base
        img_path = "preprocesamiento_individual/3000x3000.jpg"
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"No se pudo cargar la imagen en {img_path}")
        img_conexiones = img.copy()
        
        # Generar colores distintos para cada conexión
        def generar_colores_distintos(n):
            colores = []
            for i in range(n):
                h = i / n  # Distribuir en el espectro HSV
                s = 0.8 + random.uniform(-0.2, 0.2)
                v = 0.9 + random.uniform(-0.2, 0.2)
                rgb = hsv_to_rgb((h, s, v))
                bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
                colores.append(bgr)
            return colores
        
        num_conexiones = conexiones_df["conexion_id"].nunique()
        colores = generar_colores_distintos(num_conexiones)
        mapa_colores = {cid: colores[i] for i, cid in enumerate(sorted(conexiones_df["conexion_id"].unique()))}
        
        # Dibujar conexiones
        for conexion_id, grupo in conexiones_df.groupby("conexion_id"):
            color = mapa_colores[conexion_id]
            puntos = list(zip(grupo["x"].astype(int), grupo["y"].astype(int)))
            
            # Dibujar líneas entre todos los puntos
            for i in range(len(puntos)):
                for j in range(i + 1, len(puntos)):
                    pt1 = (puntos[i][0], puntos[i][1])
                    pt2 = (puntos[j][0], puntos[j][1])
                    cv2.line(img_conexiones, pt1, pt2, color, 2, cv2.LINE_AA)
            
            # Dibujar puntos
            for x, y in puntos:
                cv2.circle(img_conexiones, (x, y), 5, color, -1)
                cv2.circle(img_conexiones, (x, y), 5, (0, 0, 0), 1)  # Borde negro para resaltar
                
                # Añadir etiqueta con componente
                texto = f"{grupo.iloc[puntos.index((x, y))]['tipo_componente']}_{int(grupo.iloc[puntos.index((x, y))]['componente_id'])}:{grupo.iloc[puntos.index((x, y))]['pin_num']}"
                cv2.putText(img_conexiones, texto, (x + 10, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Añadir leyenda
        h, w = img_conexiones.shape[:2]
        margen_adicional = 100 + 30 * ((num_conexiones + 4) // 5)  # Espacio para leyenda
        img_con_leyenda = np.ones((h + margen_adicional, w, 3), dtype=np.uint8) * 255
        img_con_leyenda[:h, :w] = img_conexiones
        
        # Título de la leyenda
        cv2.putText(img_con_leyenda, "LEYENDA DE CONEXIONES", (10, h + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Dibujar leyenda
        for i, conexion_id in enumerate(sorted(mapa_colores.keys())):
            col = i % 5
            fila = i // 5
            x_pos = 10 + col * 200
            y_pos = h + 60 + fila * 30
            color = mapa_colores[conexion_id]
            
            cv2.rectangle(img_con_leyenda, (x_pos, y_pos - 10), (x_pos + 20, y_pos + 5), color, -1)
            texto = f"Conexión #{conexion_id}"
            if conexiones_df[conexiones_df["conexion_id"] == conexion_id]["zona_power"].iloc[0] is not None:
                texto += f" ({conexiones_df[conexiones_df['conexion_id'] == conexion_id]['zona_power'].iloc[0]})"
            cv2.putText(img_con_leyenda, texto, (x_pos + 30, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Guardar imagen
        output_path = "preprocesamiento_individual/conexiones_encontradas.png"
        cv2.imwrite(output_path, img_con_leyenda)
        print(f"\n✅ Imagen de conexiones guardada en: {output_path}")
        print(f"📊 Visualizadas {num_conexiones} conexiones con {len(conexiones_df)} puntos")
        
        return True
    
    except Exception as e:
        print(f"❌ Error al visualizar conexiones: {str(e)}")
        return False    
    
def limpiar_conexiones(conexiones_df):
    """
    Limpia los cables de conexiones_df, unifica conexiones basadas en cables y elimina cables.
    Guarda el DataFrame limpio en processing_state['conexiones_limpio_df'].
    
    Args:
        conexiones_df (pd.DataFrame): DataFrame con conexiones (conexion_id, componente_id, etc.).
    
    Returns:
        pd.DataFrame: DataFrame limpio sin cables.
    """
    global processing_state
    
    try:
        # Paso 1: Eliminar columnas innecesarias
        df_ajustado = conexiones_df.drop(columns=['y', 'carril_id', 'zona_power'], errors='ignore')
        
        # Paso 2: Crear lista fija de IDs de cables
        lista_cables = df_ajustado[df_ajustado['tipo_componente'] == 'cable']['componente_id'].unique()
        print(f"\n📊 Encontrados {len(lista_cables)} cables para procesar")
        
        # Paso 3: Unificar conexiones basadas en cables
        for cable_id in lista_cables:
            grupo = df_ajustado[df_ajustado['componente_id'] == cable_id]
            conexiones = grupo['conexion_id'].unique()
            
            if len(conexiones) == 2:
                menor = min(conexiones)
                mayor = max(conexiones)
                df_ajustado.loc[df_ajustado['conexion_id'] == mayor, 'conexion_id'] = menor
                print(f"✅ Unificado cable {cable_id}: conexión {mayor} → {menor}")
        
        # Paso 4: Eliminar cables y resetear índice
        df_limpio = df_ajustado[df_ajustado['tipo_componente'] != 'cable'].reset_index(drop=True)
        print(f"\n📊 DataFrame limpio: {len(df_limpio)} puntos (sin cables)")
        
        # Imprimir DataFrame limpio
        print("\n📋 DataFrame limpio (sin cables) antes de generar JSON:")
        print(df_limpio.to_string())
        
        # Guardar DataFrame limpio en processing_state
        processing_state['conexiones_limpio_df'] = df_limpio
        
        return df_limpio
    
    except Exception as e:
        print(f"\n❌ Error al limpiar conexiones: {str(e)}")
        df_limpio = pd.DataFrame(columns=[
            "conexion_id", "componente_id", "tipo_componente", "pin_num", "pin_nombre", "x"
        ])
        processing_state['conexiones_limpio_df'] = df_limpio
        return df_limpio
    
def generar_json_desde_df(df: pd.DataFrame, output_path: str = "preprocesamiento_individual/schematic_data.json") -> list:
    """
    Genera un JSON con la información de componentes y sus pines a partir de un DataFrame limpio.
    Escribe el resultado en un archivo JSON. Para botones, LEDs y resistencias, siempre incluye
    dos pines (1 y 2), marcando los no detectados como "connected": false.

    Args:
        df (pd.DataFrame): DataFrame limpio con columnas ['componente_id', 'tipo_componente', 'pin_num', 'pin_nombre', 'x', 'conexion_id']
        output_path (str): Ruta donde se guardará el archivo JSON.

    Returns:
        list: Lista de diccionarios con la información de los componentes.
    """
    global processing_state
    try:
        # Cargar netlist para nombres de pines de DIPs
        with open("componentes_dip_netlist.json", "r") as f:
            base_pins = json.load(f)

        components = []

        # Agrupar por tipo de componente y ID
        for (tipo, inst_id_raw), grupo in df.groupby(["tipo_componente", "componente_id"]):
            inst_id = int(inst_id_raw)

            # Obtener pines únicos del DataFrame
            pins_csv = set()
            for val in grupo["pin_num"].unique():
                try:
                    # Convertir a entero solo si es posible
                    pins_csv.add(int(val))
                except (ValueError, TypeError):
                    # Si no es numérico, usar como string más tarde
                    pins_csv.add(str(val))

            # Obtener pines del netlist (para DIPs) o definir para botones, LEDs y resistencias
            pins_base = set()
            if tipo in ["boton", "led", "resistencia"]:
                # Forzar dos pines para botones, LEDs y resistencias
                pins_base = {1, 2}
            else:
                # Para DIPs, usar netlist
                for k in base_pins.get(tipo, {}).keys():
                    try:
                        if k != "pinSideCount":  # Excluir pinSideCount
                            pins_base.add(int(k))
                    except ValueError:
                        continue

            # Combinar pines (usar strings para comparar)
            all_pins = sorted(pins_csv | pins_base, key=lambda x: int(x) if str(x).isdigit() else x)

            pin_list = []

            for pin_num in all_pins:
                pin_str = str(pin_num)

                # Obtener nombre del pin
                if tipo in base_pins and pin_str in base_pins[tipo]:
                    pin_name = base_pins[tipo][pin_str]
                else:
                    # Para botones, LEDs, resistencias y otros, usar nombre del DataFrame o "pin" por defecto
                    match = grupo[grupo["pin_num"].astype(str) == pin_str]
                    pin_name = match["pin_nombre"].iloc[0] if not match.empty else "pin"

                # Información completa del pin
                match = grupo[grupo["pin_num"].astype(str) == pin_str]
                if not match.empty:
                    net_raw = match["conexion_id"].iloc[0]
                    x_raw = match["x"].iloc[0]

                    # Asegurar que net_raw y x_raw sean serializables
                    net = int(net_raw) if pd.notnull(net_raw) else None
                    x = float(x_raw) if pd.notnull(x_raw) else None

                    pin_list.append({
                        "pin_num": pin_num,
                        "pin_name": pin_name,
                        "connected": True,
                        "net": net,
                        "x": x
                    })
                else:
                    pin_list.append({
                        "pin_num": pin_num,
                        "pin_name": pin_name,
                        "connected": False
                    })

            components.append({
                "type": tipo,
                "instance": inst_id,
                "pins": pin_list
            })

        # Guardar el JSON en el archivo especificado
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(components, f, indent=2)
            print(f"✅ JSON guardado en: {output_path}")
        except Exception as e:
            print(f"❌ Error al guardar JSON: {str(e)}")
            raise

        # Almacenar el JSON en processing_state
        processing_state['schematic_data'] = components
        print("✅ JSON almacenado en processing_state['schematic_data']")

        return components

    except Exception as e:
        print(f"❌ Error en generar_json_desde_df: {str(e)}")
        raise


# --- Funciones existentes (sin cambios) ---
def aplicar_clahe(img_rgb):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def aplicar_contraste_gimp(img, contraste):
    f = (259 * (contraste + 255)) / (255 * (259 - contraste))
    lut = np.array([np.clip(f * (i - 128) + 128, 0, 255) for i in range(256)]).astype(np.uint8)
    return cv2.LUT(img, lut)

def encontrar_puntos_extremos(contorno):
    if len(contorno) < 2:
        return None
    puntos = contorno.reshape(-1, 2)
    distancias = squareform(pdist(puntos))
    i, j = np.unravel_index(np.argmax(distancias), distancias.shape)
    return (tuple(puntos[i]), tuple(puntos[j]))

def generar_imagen_mascaras(imagen_base, dips):
    imagen_mascaras = imagen_base.copy()
    colores = [
        (0, 165, 255),   # Naranja
        (0, 255, 0),     # Verde
        (255, 255, 0),   # Celeste
        (0, 255, 255),   # Amarillo
        (180, 105, 255), # Rosa
        (208, 224, 64),  # Turquesa
        (219, 112, 147), # Lila
        (255, 255, 255)  # Blanco
    ]
    for idx, (i, box) in enumerate(dips):
        x1, y1, x2, y2 = map(int, box)
        color = colores[idx % len(colores)]
        cv2.rectangle(imagen_mascaras, (x1, y1), (x2, y2), color, 24)
        cv2.putText(
            imagen_mascaras,
            f"DIP{idx}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2
        )
    return imagen_mascaras
def procesar_resistencias(base_img, resistencias, modelo_patitas):
    """
    Procesa las detecciones de resistencias, genera un DataFrame con sus extremos y patitas.
    
    Args:
        base_img (np.ndarray): Imagen base (3000x3000).
        resistencias (list): Lista de tuplas (índice, caja) de detecciones de resistencias.
        modelo_patitas (YOLO): Modelo para detectar patitas.
    
    Returns:
        pd.DataFrame: DataFrame con resistencias (id, extremo1_x, extremo1_y, extremo2_x, extremo2_y, num_patitas, tipo).
    """
    try:
        # Filtro para eliminar boxes incrustados
        resistencias_filtradas = []
        resistencias_indices = set()
        for i, (idx1, box1) in enumerate(resistencias):
            x1_1, y1_1, x2_1, y2_1 = box1
            incrustado = False
            for j, (idx2, box2) in enumerate(resistencias):
                if i != j:  # Evitar comparar un box consigo mismo
                    x1_2, y1_2, x2_2, y2_2 = box2
                    # Verificar si box1 está completamente dentro de box2
                    if (x1_2 <= x1_1 and y1_2 <= y1_1 and x2_2 >= x2_1 and y2_2 >= y2_1):
                        # Calcular áreas para determinar cuál es más grande
                        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
                        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
                        if area1 < area2:
                            incrustado = True
                            break
            if not incrustado:
                resistencias_filtradas.append((idx1, box1))
                resistencias_indices.add(idx1)

        print(f"✅ Se detectaron {len(resistencias)} resistencias iniciales, filtradas a {len(resistencias_filtradas)} tras eliminar incrustados.")
        resistencias_data = []

        for idx, (i, box) in enumerate(resistencias_filtradas):
            x1, y1, x2, y2 = box
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, base_img.shape[1]), min(y2, base_img.shape[0])
            
            recorte = base_img[y1:y2, x1:x2]
            recorte_contraste = aplicar_contraste_gimp(recorte, 80)
            
            results_leads = modelo_patitas.predict(source=recorte_contraste, conf=0.4, save=False)
            masks = results_leads[0].masks.data.cpu().numpy() if results_leads[0].masks is not None else []

            # Visualizar patitas detectadas
            visualizar_patitas_componente(recorte_contraste, masks, base_output_dir, "resistencia", idx)
            
            patitas_info = []
            if len(masks) == 2:
                print(f"✅ Resistencia {idx}: Caso optimista - Detectadas 2 patitas")
                for j, mask in enumerate(masks):
                    mask_bin = (mask > 0.5).astype(np.uint8)
                    mask_resized = cv2.resize(mask_bin, (recorte.shape[1], recorte.shape[0]), 
                                            interpolation=cv2.INTER_NEAREST)
                    contornos, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contornos:
                        contorno_principal = max(contornos, key=cv2.contourArea)
                        extremos = encontrar_puntos_extremos(contorno_principal)
                        if extremos:
                            extremos_globales = [(x1 + p[0], y1 + p[1]) for p in extremos]
                            patitas_info.append({
                                'indice': j,
                                'extremos_locales': extremos,
                                'extremos_globales': extremos_globales
                            })
            
            elif len(masks) == 1:
                print(f"⚠️ Resistencia {idx}: Caso medio - Detectada 1 patita")
            elif len(masks) > 2:
                print(f"✅ Resistencia {idx}: Caso múltiple - Detectadas {len(masks)} patitas")
                for j, mask in enumerate(masks):
                    mask_bin = (mask > 0.5).astype(np.uint8)
                    mask_resized = cv2.resize(mask_bin, (recorte.shape[1], recorte.shape[0]), 
                                            interpolation=cv2.INTER_NEAREST)
                    contornos, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contornos:
                        contorno_principal = max(contornos, key=cv2.contourArea)
                        extremos = encontrar_puntos_extremos(contorno_principal)
                        if extremos:
                            extremos_globales = [(x1 + p[0], y1 + p[1]) for p in extremos]
                            patitas_info.append({
                                'indice': j,
                                'extremos_locales': extremos,
                                'extremos_globales': extremos_globales
                            })
            
            else:
                print(f"❌ Resistencia {idx}: Caso extremo - No se detectaron patitas")
            
            resistencias_data.append({
                'id': idx,
                'box': box,
                'patitas': patitas_info,
                'num_patitas': len(masks)
            })

            if len(patitas_info) == 2:
                puntos = patitas_info[0]['extremos_globales'] + patitas_info[1]['extremos_globales']
                puntos_np = np.array(puntos)
                distancias = squareform(pdist(puntos_np))
                i, j = np.unravel_index(np.argmax(distancias), distancias.shape)
                extremo1 = tuple(puntos_np[i])
                extremo2 = tuple(puntos_np[j])
                resistencias_data[-1]["extremo1"] = extremo1
                resistencias_data[-1]["extremo2"] = extremo2
            elif len(patitas_info) > 2:
                # Recolectar todos los puntos extremos globales
                puntos = []
                for info in patitas_info:
                    puntos.extend(info['extremos_globales'])
                puntos_np = np.array(puntos)
                if len(puntos_np) >= 2:
                    distancias = squareform(pdist(puntos_np))
                    i, j = np.unravel_index(np.argmax(distancias), distancias.shape)
                    extremo1 = tuple(puntos_np[i])
                    extremo2 = tuple(puntos_np[j])
                else:
                    extremo1 = (None, None)
                    extremo2 = (None, None)
                resistencias_data[-1]["extremo1"] = extremo1
                resistencias_data[-1]["extremo2"] = extremo2

        resistencias_df = pd.DataFrame([
            {
                "id": r["id"],
                "extremo1_x": r.get("extremo1", (None, None))[0],
                "extremo1_y": r.get("extremo1", (None, None))[1],
                "extremo2_x": r.get("extremo2", (None, None))[0],
                "extremo2_y": r.get("extremo2", (None, None))[1],
                "num_patitas": r["num_patitas"],
                "tipo": "resistencia"
            }
            for r in resistencias_data
        ])
        return resistencias_df

    except Exception as e:
        print(f"❌ Error al procesar resistencias: {str(e)}")
        return pd.DataFrame(columns=["id", "extremo1_x", "extremo1_y", "extremo2_x", "extremo2_y", "num_patitas", "tipo"])

def procesar_leds(base_img, leds, modelo_patitas):
    """
    Procesa las detecciones de LEDs, genera un DataFrame con sus extremos y patitas.
    
    Args:
        base_img (np.ndarray): Imagen base (3000x3000).
        leds (list): Lista de tuplas (índice, caja) de detecciones de LEDs.
        modelo_patitas (YOLO): Modelo para detectar patitas.
    
    Returns:
        pd.DataFrame: DataFrame con LEDs (id, extremo1_x, extremo1_y, extremo2_x, extremo2_y, num_patitas, tipo).
    """
    try:
        # Filtro para eliminar boxes incrustados
        leds_filtradas = []
        leds_indices = set()
        for i, (idx1, box1) in enumerate(leds):
            x1_1, y1_1, x2_1, y2_1 = box1
            incrustado = False
            for j, (idx2, box2) in enumerate(leds):
                if i != j:  # Evitar comparar un box consigo mismo
                    x1_2, y1_2, x2_2, y2_2 = box2
                    # Verificar si box1 está completamente dentro de box2
                    if (x1_2 <= x1_1 and y1_2 <= y1_1 and x2_2 >= x2_1 and y2_2 >= y2_1):
                        # Calcular áreas para determinar cuál es más grande
                        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
                        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
                        if area1 < area2:
                            incrustado = True
                            break
            if not incrustado:
                leds_filtradas.append((idx1, box1))
                leds_indices.add(idx1)

        print(f"✅ Se detectaron {len(leds)} LEDs iniciales, filtradas a {len(leds_filtradas)} tras eliminar incrustados.")
        leds_data = []

        for idx, (i, box) in enumerate(leds_filtradas):
            x1, y1, x2, y2 = box
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, base_img.shape[1]), min(y2, base_img.shape[0])
            
            recorte = base_img[y1:y2, x1:x2]
            recorte_contraste = aplicar_contraste_gimp(recorte, 80)
            
            results_leads = modelo_patitas.predict(source=recorte_contraste, conf=0.4, save=False)
            masks = results_leads[0].masks.data.cpu().numpy() if results_leads[0].masks is not None else []

            # Visualizar patitas detectadas
            visualizar_patitas_componente(recorte_contraste, masks, base_output_dir, "led", idx)
            
            patitas_info = []
            if len(masks) == 2:
                print(f"✅ LED {idx}: Caso optimista - Detectadas 2 patitas")
                for j, mask in enumerate(masks):
                    mask_bin = (mask > 0.5).astype(np.uint8)
                    mask_resized = cv2.resize(mask_bin, (recorte.shape[1], recorte.shape[0]), 
                                            interpolation=cv2.INTER_NEAREST)
                    contornos, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contornos:
                        contorno_principal = max(contornos, key=cv2.contourArea)
                        extremos = encontrar_puntos_extremos(contorno_principal)
                        if extremos:
                            extremos_globales = [(x1 + p[0], y1 + p[1]) for p in extremos]
                            patitas_info.append({
                                'indice': j,
                                'extremos_locales': extremos,
                                'extremos_globales': extremos_globales
                            })
            
            elif len(masks) == 1:
                print(f"⚠️ LED {idx}: Caso medio - Detectada 1 patita")
            elif len(masks) > 2:
                print(f"✅ LED {idx}: Caso múltiple - Detectadas {len(masks)} patitas")
                for j, mask in enumerate(masks):
                    mask_bin = (mask > 0.5).astype(np.uint8)
                    mask_resized = cv2.resize(mask_bin, (recorte.shape[1], recorte.shape[0]), 
                                            interpolation=cv2.INTER_NEAREST)
                    contornos, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contornos:
                        contorno_principal = max(contornos, key=cv2.contourArea)
                        extremos = encontrar_puntos_extremos(contorno_principal)
                        if extremos:
                            extremos_globales = [(x1 + p[0], y1 + p[1]) for p in extremos]
                            patitas_info.append({
                                'indice': j,
                                'extremos_locales': extremos,
                                'extremos_globales': extremos_globales
                            })    
            else:
                print(f"❌ LED {idx}: Caso extremo - No se detectaron patitas")
            
            leds_data.append({
                'id': idx,
                'box': box,
                'patitas': patitas_info,
                'num_patitas': len(masks)
            })

            if len(patitas_info) == 2:
                puntos = patitas_info[0]['extremos_globales'] + patitas_info[1]['extremos_globales']
                puntos_np = np.array(puntos)
                distancias = squareform(pdist(puntos_np))
                i, j = np.unravel_index(np.argmax(distancias), distancias.shape)
                extremo1 = tuple(puntos_np[i])
                extremo2 = tuple(puntos_np[j])
                leds_data[-1]["extremo1"] = extremo1
                leds_data[-1]["extremo2"] = extremo2
            elif len(patitas_info) > 2:
                # Recolectar todos los puntos extremos globales
                puntos = []
                for info in patitas_info:
                    puntos.extend(info['extremos_globales'])
                puntos_np = np.array(puntos)
                if len(puntos_np) >= 2:
                    distancias = squareform(pdist(puntos_np))
                    i, j = np.unravel_index(np.argmax(distancias), distancias.shape)
                    extremo1 = tuple(puntos_np[i])
                    extremo2 = tuple(puntos_np[j])
                else:
                    extremo1 = (None, None)
                    extremo2 = (None, None)
                leds_data[-1]["extremo1"] = extremo1
                leds_data[-1]["extremo2"] = extremo2    

        leds_df = pd.DataFrame([
            {
                "id": r["id"],
                "extremo1_x": r.get("extremo1", (None, None))[0],
                "extremo1_y": r.get("extremo1", (None, None))[1],
                "extremo2_x": r.get("extremo2", (None, None))[0],
                "extremo2_y": r.get("extremo2", (None, None))[1],
                "num_patitas": r["num_patitas"],
                "tipo": "led"
            }
            for r in leds_data
        ])
        return leds_df

    except Exception as e:
        print(f"❌ Error al procesar LEDs: {str(e)}")
        return pd.DataFrame(columns=["id", "extremo1_x", "extremo1_y", "extremo2_x", "extremo2_y", "num_patitas", "tipo"])

def procesar_botones(base_img, botones):
    """
    Procesa las detecciones de botones, genera un DataFrame con sus extremos compensados.
    
    Args:
        base_img (np.ndarray): Imagen base (3000x3000).
        botones (list): Lista de tuplas (índice, caja) de detecciones de botones.
    
    Returns:
        pd.DataFrame: DataFrame con botones (id, extremo1_x, extremo1_y, extremo2_x, extremo2_y, tipo).
    """
    try:
        print(f"✅ Se detectaron {len(botones)} botones.")
        botones_data = []
        centro_img = (base_img.shape[1] // 2, base_img.shape[0] // 2)

        def compensar_centro(punto):
            dx = centro_img[0] - punto[0]
            dy = centro_img[1] - punto[1]
            factor = 0.025 * 1.2
            return (int(punto[0] + dx * factor), int(punto[1] + dy * factor))

        def compensar_fuera(punto, factor=0.025):
            dx = punto[0] - centro_img[0]
            dy = punto[1] - centro_img[1]
            return (int(punto[0] + dx * factor), int(punto[1] + dy * factor))

        for idx, (i, box) in enumerate(botones):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            if w < h:
                p1 = (cx, y1)
                p2 = (cx, y2)
                p1_corr = compensar_centro(p1)
                p2_corr = compensar_centro(p2)
            else:
                cy_uniforme = cy
                p1 = (x1, cy_uniforme)
                p2 = (x2, cy_uniforme)

                if abs(p1[0] - centro_img[0]) > abs(p2[0] - centro_img[0]):
                    p1_corr = compensar_centro(p1)
                    p2_corr = compensar_fuera(p2)
                else:
                    p1_corr = compensar_fuera(p1)
                    p2_corr = compensar_centro(p2)

                p1_corr = (p1_corr[0], cy_uniforme)
                p2_corr = (p2_corr[0], cy_uniforme)

            botones_data.append({
                "id": idx,
                "box": box,
                "punto1": p1_corr,
                "punto2": p2_corr
            })

        botones_df = pd.DataFrame([
            {
                "id": b["id"],
                "extremo1_x": b["punto1"][0],
                "extremo1_y": b["punto1"][1],
                "extremo2_x": b["punto2"][0],
                "extremo2_y": b["punto2"][1],
                "tipo": "boton"
            }
            for b in botones_data
        ])
        return botones_df

    except Exception as e:
        print(f"❌ Error al procesar botones: {str(e)}")
        return pd.DataFrame(columns=["id", "extremo1_x", "extremo1_y", "extremo2_x", "extremo2_y", "tipo"])
    
def procesar_cristales(base_img, cristales):
    """
    Procesa las detecciones de cristales, genera un DataFrame con los puntos P1 y P3 compensados.
    
    Args:
        base_img (np.ndarray): Imagen base (3000x3000).
        cristales (list): Lista de tuplas (índice, caja) de detecciones de cristales.
    
    Returns:
        pd.DataFrame: DataFrame con cristales (id, extremo1_x, extremo1_y, extremo2_x, extremo2_y, tipo).
    """
    try:
        print(f"✅ Se detectaron {len(cristales)} cristales.")
        cristales_data = []
        centro_img = (base_img.shape[1] // 2, base_img.shape[0] // 2)
        # Crear una copia de la imagen base para dibujar (para depuración)
        img_debug = base_img.copy()

        def compensar_centro(punto):
            dx = centro_img[0] - punto[0]
            dy = centro_img[1] - punto[1]
            factor = 0.025 * 0.6
            return (int(punto[0] + dx * factor), int(punto[1] + dy * factor))

        for idx, (i, box) in enumerate(cristales):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Determinar si el cristal es vertical (w < h) o horizontal (w >= h)
            if w < h:
                # Dividir la altura en 5 partes iguales
                segment_height = h / 5
                points = [
                    (cx, int(y1 + segment_height * i + segment_height / 2))
                    for i in range(5)
                ]
            else:
                # Dividir el ancho en 5 partes iguales
                segment_width = w / 5
                points = [
                    (int(x1 + segment_width * i + segment_width / 2), cy)
                    for i in range(5)
                ]

            # Aplicar compensación de perspectiva a cada punto
            points_compensados = [compensar_centro(punto) for punto in points]

            # Seleccionar P1 (índice 1) y P3 (índice 3) para los pines
            p1_corr = points_compensados[1]  # P1
            p3_corr = points_compensados[3]  # P3

            # Agregar datos al listado
            cristales_data.append({
                "id": idx,
                "punto1": p1_corr,
                "punto2": p3_corr
            })

        #     # Dibujar los puntos compensados en la imagen (para depuración)
        #     for point in points_compensados:
        #         color = (0, 255, 0)  # Verde para puntos no seleccionados
        #         if point in (p1_corr, p3_corr):
        #             color = (0, 0, 255)  # Rojo para P1 y P3
        #         cv2.circle(img_debug, point, 5, color, -1)
        #         cv2.putText(img_debug, f"P{points_compensados.index(point)}",
        #                    (point[0] + 10, point[1]),
        #                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        #     # Dibujar el bounding box para referencia
        #     cv2.rectangle(img_debug, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # # Guardar la imagen de depuración (opcional, para verificar visualmente)
        # cv2.imwrite("debug_cristales_final.jpg", img_debug)

        # Crear DataFrame con los puntos P1 y P3
        cristales_df = pd.DataFrame([
            {
                "id": c["id"],
                "extremo1_x": c["punto1"][0],
                "extremo1_y": c["punto1"][1],
                "extremo2_x": c["punto2"][0],
                "extremo2_y": c["punto2"][1],
                "tipo": "cristal"
            }
            for c in cristales_data
        ])

        return cristales_df

    except Exception as e:
        print(f"❌ Error al procesar cristales: {str(e)}")
        return pd.DataFrame(columns=["id", "extremo1_x", "extremo1_y", "extremo2_x", "extremo2_y", "tipo"])

def procesar_capacitores_ceramicos(base_img, cap_cers):
    """
    Procesa las detecciones de capacitores cerámicos, genera un DataFrame con sus extremos compensados.
    
    Args:
        base_img (np.ndarray): Imagen base (3000x3000).
        cap_cers (list): Lista de tuplas (índice, caja) de detecciones de capacitores cerámicos.
    
    Returns:
        pd.DataFrame: DataFrame con capacitores cerámicos (id, extremo1_x, extremo1_y, extremo2_x, extremo2_y, tipo).
    """
    try:
        print(f"✅ Se detectaron {len(cap_cers)} capacitores cerámicos.")
        cap_cers_data = []
        centro_img = (base_img.shape[1] // 2, base_img.shape[0] // 2)

        def compensar_centro(punto):
            dx = centro_img[0] - punto[0]
            dy = centro_img[1] - punto[1]
            factor = 0.025 * 0.3
            return (int(punto[0] + dx * factor), int(punto[1] + dy * factor))

        for idx, (i, box) in enumerate(cap_cers):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            if w < h*0.7:
                segment_height = h / 3
                points = [
                    (cx, int(y1 + segment_height * i + segment_height / 2))
                    for i in range(3)
                ]
            else:
                segment_width = w / 3
                points = [
                    (int(x1 + segment_width * i + segment_width / 2), cy)
                    for i in range(3)
                ]
            # Aplicar compensación de perspectiva a cada punto
            points_compensados = [compensar_centro(punto) for punto in points]

            # Seleccionar P1 (índice 1) y P3 (índice 3) para los pines
            p1_corr = points_compensados[0]  # P1
            p3_corr = points_compensados[2]  # P3

            # Agregar datos al listado
            cap_cers_data.append({
                "id": idx,
                "punto1": p1_corr,
                "punto2": p3_corr
            })
        # Crear DataFrame con los puntos P1 y P3
        cap_cers_df = pd.DataFrame([
            {
                "id": c["id"],
                "extremo1_x": c["punto1"][0],
                "extremo1_y": c["punto1"][1],
                "extremo2_x": c["punto2"][0],
                "extremo2_y": c["punto2"][1],
                "tipo": "cap_cer"
            }
            for c in cap_cers_data
        ])
        return cap_cers_df
    except Exception as e:
        print(f"❌ Error al procesar capacitores cerámicos: {str(e)}")
        return pd.DataFrame(columns=["id", "extremo1_x", "extremo1_y", "extremo2_x", "extremo2_y", "tipo"])
               
    
def procesar_detecciones():
    global processing_state, modelo_componentes, modelo_patitas, modelo_dip, base_output_dir
    try:
        imagen_base_path = os.path.join(base_output_dir, "3000x3000.jpg")
        input_componentes = os.path.join(base_output_dir, "componentes_800.jpg")
        escala_componentes = 3000 / 800

        base_img = cv2.imread(imagen_base_path)

        results = modelo_componentes.predict(source=input_componentes, conf=0.4, save=False)
        all_boxes, all_classes, all_confs = [], [], []

        for r in results:
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy() * escala_componentes
                xyxy = xyxy.astype(int)
                label = int(box.cls[0])
                conf = float(box.conf[0])
                all_boxes.append(xyxy)
                all_classes.append(label)
                all_confs.append(conf)

        #Visualizar detecciones para debugging
        visualizar_detecciones_componentes(base_img, all_boxes, all_classes, all_confs, 
                                         modelo_componentes.names, base_output_dir)
        

        # Filtrar detecciones por tipo de componente
        res_class_id = [k for k, v in modelo_componentes.names.items() if v == "res"][0]
        resistencias = [
            (i, box) for i, box in enumerate(all_boxes) if all_classes[i] == res_class_id
        ]

        led_class_id = [k for k, v in modelo_componentes.names.items() if v == "led"][0]
        leds = [
            (i, box) for i, box in enumerate(all_boxes) if all_classes[i] == led_class_id
        ]

        button_class_id = [k for k, v in modelo_componentes.names.items() if v == "button"][0]
        botones = [
            (i, box) for i, box in enumerate(all_boxes) if all_classes[i] == button_class_id
        ]

        xtal_class_id = [k for k, v in modelo_componentes.names.items() if v == "xtal"][0]
        cristals = [
            (i, box) for i, box in enumerate(all_boxes) if all_classes[i] == xtal_class_id
        ]

        cap_cer_class_id = [k for k, v in modelo_componentes.names.items() if v == "cap_cer"][0]
        cap_cers = [
            (i, box) for i, box in enumerate(all_boxes) if all_classes[i] == cap_cer_class_id
        ]

        # Procesar resistencias, LEDs y botones
        resistencias_df = procesar_resistencias(base_img, resistencias, modelo_patitas)
        leds_df = procesar_leds(base_img, leds, modelo_patitas)
        botones_df = procesar_botones(base_img, botones)
        cristales_df = procesar_cristales(base_img, cristals)
        cap_cers_df = procesar_capacitores_ceramicos(base_img, cap_cers)
        

        # Procesar cables
        img_cables_path = os.path.join(base_output_dir, "cables_1024.jpg")
        img_puntas_path = os.path.join(base_output_dir, "puntas_cruces.jpg")
        cables_df = procesar_cables_y_puntas(img_puntas_path, img_puntas_path)

        # Guardar DataFrames en estado global
        processing_state['resistencias_df'] = resistencias_df
        processing_state['leds_df'] = leds_df
        processing_state['botones_df'] = botones_df
        processing_state['cables_df'] = cables_df
        processing_state['cristales_df'] = cristales_df
        processing_state['cap_cers_df'] = cap_cers_df

        print("📊 Resistencias DataFrame:")
        print(resistencias_df.to_string())
        print("📊 LEDs DataFrame:")
        print(leds_df.to_string())
        print("📊 Botones DataFrame:")
        print(botones_df.to_string())
        print("📊 Cristales DataFrame:")
        print(cristales_df.to_string())
        print("📊 Capacitores Cerámicos DataFrame:")
        print(cap_cers_df.to_string())
        print("📊 Cables DataFrame:")
        print(cables_df.to_string())

        # Procesar Ultrasónicos (usonic)
        usonic_class_id = [k for k, v in modelo_componentes.names.items() if v == "usonic"][0]
        usonics = [(i, box) for i, box in enumerate(all_boxes) if all_classes[i] == usonic_class_id]
        print(f"✅ Se detectaron {len(usonics)} componentes ultrasónicos (usonic) antes del filtrado.")

        # Filtrado de detecciones solapadas usando IoU
        if len(usonics) > 1:  # Solo filtrar si hay más de una detección
            filtered_usonics = []
            boxes = [(idx, box) for idx, box in usonics]  # Mantener índices originales
            while boxes:
                current_idx, current_box = boxes.pop(0)
                x1_a, y1_a, x2_a, y2_a = map(int, current_box)
                area_a = (x2_a - x1_a) * (y2_a - y1_a)
                keep = True
                i = 0
                while i < len(boxes):
                    idx_b, box_b = boxes[i]
                    x1_b, y1_b, x2_b, y2_b = map(int, box_b)
                    area_b = (x2_b - x1_b) * (y2_b - y1_b)

                    # Calcular intersección
                    x1_inter = max(x1_a, x1_b)
                    y1_inter = max(y1_a, y1_b)
                    x2_inter = min(x2_a, x2_b)
                    y2_inter = min(y2_a, y2_b)
                    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

                    # Calcular unión
                    union_area = area_a + area_b - inter_area
                    iou = inter_area / union_area if union_area > 0 else 0

                    if iou >= 0.5:  # Solapamiento del 50% o más
                        if area_a > area_b:
                            boxes.pop(i)  # Descartar el box más pequeño (box_b)
                        else:
                            keep = False  # Descartar el box actual (current_box)
                            break
                    else:
                        i += 1

                if keep:
                    filtered_usonics.append((current_idx, current_box))

            usonics = filtered_usonics
            print(f"✅ Se detectaron {len(usonics)} componentes ultrasónicos (usonic) después del filtrado.")

        if usonics:
            usonic_detections = []
            colores = [
                (255, 165, 0),   # Naranja
                (0, 255, 0),     # Verde
                (0, 255, 255),   # Celeste
                (255, 255, 0),   # Amarillo
                (255, 105, 180), # Rosa
                (64, 224, 208),  # Turquesa
                (147, 112, 219), # Lila
                (255, 255, 255)  # Blanco
            ]
            for idx, (i, box) in enumerate(usonics):
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, base_img.shape[1]), min(y2, base_img.shape[0])

                # Reducir el margen del bounding box un 10% por lado (ajustado a 0.1 como indicaste)
                width = x2 - x1
                height = y2 - y1
                margin_x = int(0.1 * width)  # 10% del ancho
                x1_margen = x1 + margin_x
                x2_margen = x2 - margin_x
                y1_margen = y1  # Mantener y1 sin cambio por ahora
                y2_margen = y2  # Mantener y2 sin cambio por ahora

                # Asegurar que el nuevo recorte esté dentro de los límites de la imagen
                x1_margen = max(x1_margen, 0)
                x2_margen = min(x2_margen, base_img.shape[1])
                y1_margen = max(y1_margen, 0)
                y2_margen = min(y2_margen, base_img.shape[0])

                recorte = base_img[y1_margen:y2_margen, x1_margen:x2_margen]
                pins = []

                # Aplicar el modelo de detección de pines (usando modelo_usonic)
                results = modelo_usonic.predict(source=recorte, conf=0.4, save=False)
                if results[0].boxes is not None:
                    for box_pin in results[0].boxes:
                        clase = int(box_pin.cls[0])
                        conf = float(box_pin.conf[0])
                        nombre = "pin"  # Solo una clase, "pin"
                        x1_pin, y1_pin, x2_pin, y2_pin = box_pin.xyxy[0].cpu().numpy().astype(int)
                        x1g, y1g = x1_margen + x1_pin, y1_margen + y1_pin
                        x2g, y2g = x1_margen + x2_pin, y1_margen + y2_pin
                        centro_x = (x1g + x2g) // 2
                        centro_y = (y1g + y2g) // 2
                        print(f"📌 Debugging - Detección en usonic {idx}: Clase={nombre}, Confianza={conf:.2f}, Centro=(x={centro_x}, y={centro_y})")
                        if clase == 0:  # "pin" es la clase 0 en modelo_usonic
                            pin_info = {"x": int(centro_x), "y": int(centro_y)}
                            pins.append(pin_info)
                print(f"📌 Debugging - Pines detectados para usonic {idx}: {pins}")

                # Numeración de pines
                if pins:
                    y_medio = (y1_margen + y2_margen) / 2
                    pin_representativo = pins[0]  # Usamos el primer pin para decidir la mitad
                    if pin_representativo["y"] > y_medio:
                        # Pines en la mitad superior, numerar de derecha a izquierda
                        pins_sorted = sorted(pins, key=lambda p: p["x"], reverse=True)
                        pin_numbers = list(range(1, len(pins) + 1))
                    else:
                        # Pines en la mitad inferior, numerar de izquierda a derecha
                        pins_sorted = sorted(pins, key=lambda p: p["x"])
                        pin_numbers = list(range(1, len(pins) + 1))
                    # Asignar números a los pines
                    pins_with_numbers = [{"x": p["x"], "y": p["y"], "number": num} for p, num in zip(pins_sorted, pin_numbers)]
                else:
                    pins_with_numbers = []

                # Debugging: Guardar imagen recortada con bounding box y pines numerados
                output_dir = os.path.join(base_output_dir, "usonics")
                os.makedirs(output_dir, exist_ok=True)
                debug_img = recorte.copy()
                cv2.rectangle(debug_img, (0, 0), (x2_margen - x1_margen, y2_margen - y1_margen), (0, 255, 0), 2)  # Bounding box en verde
                for pin in pins_with_numbers:
                    x_local = pin["x"] - x1_margen
                    y_local = pin["y"] - y1_margen
                    if 0 <= x_local < x2_margen - x1_margen and 0 <= y_local < y2_margen - y1_margen:
                        cv2.circle(debug_img, (x_local, y_local), 2, (0, 0, 255), -1)  # Círculo rojo
                        cv2.putText(debug_img, str(pin["number"]), (x_local + 5, y_local - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                output_path = os.path.join(output_dir, f"usonic_{idx}_detection.jpg")
                cv2.imwrite(output_path, debug_img)
                print(f"📌 Debugging - Guardada imagen de ultrasónico {idx} con pines en {output_path}")

                # Guardar imagen recortada solo con el ultrasónico (sin pines)
                debug_img_no_pins = recorte.copy()
                cv2.rectangle(debug_img_no_pins, (0, 0), (x2_margen - x1_margen, y2_margen - y1_margen), (0, 255, 0), 2)  # Bounding box en verde
                output_path_no_pins = os.path.join(output_dir, f"usonic_{idx}_detection_no_pins.jpg")
                cv2.imwrite(output_path_no_pins, debug_img_no_pins)
                print(f"📌 Debugging - Guardada imagen de ultrasónico {idx} sin pines en {output_path_no_pins}")

                usonic_detections.append({
                    "id": idx,
                    "box": [int(x1), int(y1), int(x2), int(y2)],  # Bounding box original
                    "box_reduced": [int(x1_margen), int(y1_margen), int(x2_margen), int(y2_margen)],  # Bounding box reducido
                    "pins": pins_with_numbers,  # Pines con números asignados
                    "color": list(colores[idx % len(colores)])
                })

            # Almacenar en processing_state
            if usonics:
                processing_state['usonic_detections'] = usonic_detections
                # Crear DataFrame
                usonic_df = pd.DataFrame([
                    {
                        "id": usonic["id"],
                        "box": usonic["box"],
                        "points": 0,  # Asumimos que no hay points para usonic
                        "pins": [{"x": p["x"], "y": p["y"], "number": p["number"]} for p in usonic["pins"]]
                    }
                    for usonic in usonic_detections
                ])
                processing_state['usonic_df'] = usonic_df
                for usonic in usonic_detections:
                    print(f"\n🧩 Ultrasónico {usonic['id']} (Box original: {usonic['box']}, Box reducido: {usonic['box_reduced']}, Color: {usonic['color']}):")
                    print(f"  - Pines detectados: {len(usonic['pins'])}")
                    print(f"  - Pines con números: {[(p['x'], p['y'], p['number']) for p in usonic['pins']]}")

                
        # Procesar DIPs
        dip_class_id = [k for k, v in modelo_componentes.names.items() if v == "dip"][0]
        dips = [(i, box) for i, box in enumerate(all_boxes) if all_classes[i] == dip_class_id]
        print(f"✅ Se detectaron {len(dips)} componentes DIP.")

        if dips:
            dip_detecciones_detalladas = []
            clases_dip = {0: "muesca", 1: "pins", 2: "point"}
            colores = [
                (255, 165, 0),   # Naranja
                (0, 255, 0),     # Verde
                (0, 255, 255),   # Celeste
                (255, 255, 0),   # Amarillo
                (255, 105, 180), # Rosa
                (64, 224, 208),  # Turquesa
                (147, 112, 219), # Lila
                (255, 255, 255)  # Blanco
            ]
            for idx, (i, box) in enumerate(dips):
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, base_img.shape[1]), min(y2, base_img.shape[0])

                # Detección de muescas en el box original
                box_original = base_img[y1:y2, x1:x2]
                muescas = []  # Reiniciar muescas para usar solo las del box original
                results_original = modelo_dip.predict(source=box_original, conf=0.4, save=False)
                if results_original[0].boxes is not None:
                    for box_pin in results_original[0].boxes:
                        clase = int(box_pin.cls[0])
                        if clases_dip.get(clase) == "muesca":
                            x1_pin, y1_pin, x2_pin, y2_pin = box_pin.xyxy[0].cpu().numpy().astype(int)
                            x1g, y1g = x1 + x1_pin, y1 + y1_pin
                            x2g, y2g = x1 + x2_pin, y1 + y2_pin
                            muescas.append([int(x1g), int(y1g), int(x2g), int(y2g)])

                # Calcular margen del 5% del ancho
                margen_x = int(0.04 * (x2 - x1))
                x1_margen = x1 + margen_x
                x2_margen = x2 - margen_x

                # Asegurar que el nuevo recorte esté dentro de los límites de la imagen
                x1_margen = max(x1_margen, 0)
                x2_margen = min(x2_margen, base_img.shape[1])
                y1_margen = max(y1, 0)
                y2_margen = min(y2, base_img.shape[0])

                recorte = base_img[y1_margen:y2_margen, x1_margen:x2_margen]
                points, pins_superior, pins_inferior = [], [], []

                # Detección en el recorte (ignorar muescas del recorte)
                results = modelo_dip.predict(source=recorte, conf=0.4, save=False)
                if results[0].boxes is not None:
                    for box_pin in results[0].boxes:
                        clase = int(box_pin.cls[0])
                        nombre = clases_dip.get(clase, f"desconocido_{clase}")
                        x1_pin, y1_pin, x2_pin, y2_pin = box_pin.xyxy[0].cpu().numpy().astype(int)
                        x1g, y1g = x1_margen + x1_pin, y1_margen + y1_pin
                        x2g, y2g = x1_margen + x2_pin, y1_margen + y2_pin
                        centro_x = (x1g + x2g) // 2
                        centro_y = (y1g + y2g) // 2

                        if nombre == "point":
                            points.append([int(x1g), int(y1g), int(x2g), int(y2g)])
                        elif nombre == "pins":
                            umbral_y = (y1_margen + y2_margen) / 2
                            pin_info = {"x": int(centro_x), "y": int(centro_y)}
                            if centro_y < umbral_y:
                                pins_superior.append(pin_info)
                            else:
                                pins_inferior.append(pin_info)

                prom_y_sup = float(np.mean([p['y'] for p in pins_superior])) if pins_superior else None
                prom_y_inf = float(np.mean([p['y'] for p in pins_inferior])) if pins_inferior else None
                box_coords = [int(x) for x in box]

                dip_detecciones_detalladas.append({
                    "id": idx,
                    "box": box_coords,
                    "muescas": muescas,  # Usar muescas del box original
                    "points": points,
                    "pins_superior": pins_superior,
                    "pins_inferior": pins_inferior,
                    "prom_y_sup": prom_y_sup,
                    "prom_y_inf": prom_y_inf,
                    "color": list(colores[idx % len(colores)])  # Añadir color como [R, G, B]
                })

                # Debugging: Guardar imagen recortada con bounding box y pines
                output_dir = os.path.join(base_output_dir, "dips")
                os.makedirs(output_dir, exist_ok=True)
                debug_img = recorte.copy()
                cv2.rectangle(debug_img, (0, 0), (x2_margen - x1_margen, y2_margen - y1_margen), (0, 255, 0), 2)  # Bounding box en verde

                # Dibujar pines superiores (rojo)
                for pin in pins_superior:
                    x_local = pin['x'] - x1_margen
                    y_local = pin['y'] - y1_margen
                    if 0 <= x_local < x2_margen - x1_margen and 0 <= y_local < y2_margen - y1_margen:
                        cv2.circle(debug_img, (x_local, y_local), 2, (0, 0, 255), -1)  # Círculo rojo

                # Dibujar pines inferiores (azul)
                for pin in pins_inferior:
                    x_local = pin['x'] - x1_margen
                    y_local = pin['y'] - y1_margen
                    if 0 <= x_local < x2_margen - x1_margen and 0 <= y_local < y2_margen - y1_margen:
                        cv2.circle(debug_img, (x_local, y_local), 2, (255, 0, 0), -1)  # Círculo azul

                output_path = os.path.join(output_dir, f"dip_{idx}_detection.jpg")
                cv2.imwrite(output_path, debug_img)
                print(f"📌 Debugging - Guardada imagen de DIP {idx} con pines en {output_path}")

            img_mascaras = generar_imagen_mascaras(base_img, dips)
            _, buffer = cv2.imencode('.jpg', img_mascaras)
            dip_image_base64 = base64.b64encode(buffer).decode('utf-8')

            processing_state['dip_detections'] = dip_detecciones_detalladas
            processing_state['dip_image_base64'] = dip_image_base64
            processing_state['dip_selection_pending'] = True
            processing_state['is_processing'] = False

            for dip in dip_detecciones_detalladas:
                print(f"\n🧩 DIP {dip['id']} (Box: {dip['box']}, Color: {dip['color']}):")
                print(f"  - Muescas detectadas: {len(dip['muescas'])}")
                print(f"  - Points detectados: {len(dip['points'])}")
                print(f"  - Pins superior: {len(dip['pins_superior'])} | Promedio Y: {dip['prom_y_sup']:.2f}" if dip['prom_y_sup'] else "  - Pins superior: 0")
                print(f"  - Pins inferior: {len(dip['pins_inferior'])} | Promedio Y: {dip['prom_y_inf']:.2f}" if dip['prom_y_inf'] else "  - Pins inferior: 0")
                print(f"  - X de pins superior: {[p['x'] for p in dip['pins_superior']]}")
                print(f"  - X de pins inferior: {[p['x'] for p in dip['pins_inferior']]}")

            print("⏳ Esperando selecciones de DIPs desde la app...")
            return
        # Generar imagen sin DIPs
        generar_imagen_componentes(base_img, resistencias_df, leds_df, botones_df, cables_df,cristales_df)

        unificar_dataframes(resistencias_df, leds_df, botones_df, cables_df)
        
        generar_conexiones_carriles(processing_state['unificado_df'])

        visualizar_conexiones(processing_state['conexiones_df'])

        # Limpiar conexiones
        df_limpio = limpiar_conexiones(processing_state['conexiones_df'])
        # Generar JSON
        # generar_json_schematic(df_limpio)
        generar_json_desde_df(df_limpio)

        processing_state['dips_df'] = pd.DataFrame()
        processing_state['detections_completed'] = True
        processing_state['is_processing'] = False

        print("✅ Detecciones completadas.")
        print("📊 Resistencias DataFrame:")
        print(resistencias_df.to_string())
        print("📊 LEDs DataFrame:")
        print(leds_df.to_string())
        print("📊 Botones DataFrame:")
        print(botones_df.to_string())
        print("📊 DIPs DataFrame: (Vacío)")

    except Exception as e:
        print(f"❌ Error en las detecciones: {str(e)}")
        processing_state['is_processing'] = False
        processing_state['detections_completed'] = False
        processing_state['dip_selection_pending'] = False
        processing_state['resistencias_df'] = None
        processing_state['leds_df'] = None
        processing_state['botones_df'] = None
        processing_state['dips_df'] = None
        processing_state['dip_detections'] = None
        processing_state['dip_image_base64'] = None
        processing_state['dip_selections'] = None

@app.route('/procesar-imagen', methods=['POST'])
def procesar_imagen():
    data = request.get_json()

    if 'imagen_base64' not in data:
        return jsonify({'error': 'Falta la clave "imagen_base64"'}), 400

    # Enviar respuesta inmediata
    response = {
        'mensaje': 'Imagen recibida, procesamiento iniciado'
    }

    # Procesar imagen en un hilo separado
    def procesar_imagen_fondo(img_data):
        try:
            np_arr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            img_resized = cv2.resize(img, (3000, 3000), interpolation=cv2.INTER_LINEAR)
            img_clahe = aplicar_clahe(img_resized)
            img_dark = cv2.convertScaleAbs(img_clahe, alpha=0.95, beta=0)
            img_final = cv2.GaussianBlur(img_dark, (3, 3), 0)

            output_3000 = os.path.join(base_output_dir, "3000x3000.jpg")
            cv2.imwrite(output_3000, img_final)

            exportaciones = [
                ("zona_640", 640),
                ("componentes_800", 800),
                ("cables_1024", 1024)
            ]
            
            output_paths = [output_3000]
            for nombre_archivo, tamano in exportaciones:
                resized = cv2.resize(img_final, (tamano, tamano), interpolation=cv2.INTER_LINEAR)
                output_path = os.path.join(base_output_dir, f"{nombre_archivo}.jpg")
                cv2.imwrite(output_path, resized)
                output_paths.append(output_path)

            # Imagen adicional 1024x1024 sin efectos
            puntas_cruces = cv2.resize(img_resized, (1024, 1024), interpolation=cv2.INTER_LINEAR)
            output_cruces = os.path.join(base_output_dir, "puntas_cruces.jpg")
            cv2.imwrite(output_cruces, puntas_cruces)
            print(f"✅ Generado: puntas_cruces.jpg (1024x1024 sin efecto)")

            # Inicializar estado
            processing_state['is_processing'] = True
            processing_state['detections_completed'] = False
            processing_state['dip_selection_pending'] = False
            processing_state['resistencias_df'] = None
            processing_state['leds_df'] = None
            processing_state['botones_df'] = None
            processing_state['cristales_df'] = None
            processing_state['cap_cers_df'] = None
            processing_state['usonic_detections'] = None
            processing_state['dips_df'] = None
            processing_state['dip_detections'] = None
            processing_state['dip_image_base64'] = None
            processing_state['dip_selections'] = None
            processing_state['unificado_df'] = None
            processing_state['conexiones_df'] = None
            processing_state['conexiones_limpio_df'] = None
            processing_state['schematic_data'] = []

            # Guardar output_paths para otros endpoints
            processing_state['output_paths'] = output_paths

            # Iniciar detecciones
            threading.Thread(target=procesar_detecciones, daemon=True).start()

        except Exception as e:
            print(f"❌ Error en procesamiento de imagen: {str(e)}")
            processing_state['is_processing'] = False
            processing_state['error'] = str(e)

    # Decodificar base64 y ejecutar en hilo
    try:
        img_data = base64.b64decode(data['imagen_base64'])
        threading.Thread(target=procesar_imagen_fondo, args=(img_data,), daemon=True).start()
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': f'Error al decodificar imagen: {str(e)}'}), 400

@app.route('/estado-detecciones', methods=['GET'])
def estado_detecciones():
    global processing_state
    if processing_state['is_processing']:
        print("DEBUG: Estado - Detecciones en curso")
        return jsonify({'mensaje': 'Detecciones en curso'})
    elif processing_state['dip_selection_pending']:
        print("DEBUG: Estado - Esperando selecciones de DIPs")
        return jsonify({
            'mensaje': 'DIPs detectados',
            'dip_image_base64': processing_state['dip_image_base64'],
            'dip_detections': processing_state['dip_detections'],
            'dip_netlist': dip_netlist
        })
    elif processing_state['detections_completed']:
        print(f"DEBUG: Estado - Detecciones completadas, schematic_data contiene {len(processing_state['schematic_data'])} componentes")
        return jsonify({
            'mensaje': 'Detecciones completadas con éxito',
            "schematic_data": processing_state['schematic_data']
        })
    else:
        print("DEBUG: Estado - Sin detecciones")
        return jsonify({'mensaje': 'No hay detecciones en curso o completadas'})

@app.route('/dip-selections', methods=['POST'])
def dip_selections():
    data = request.get_json()
    print(f"📥 JSON recibido en /dip-selections: {data}")

    if not data or 'selections' not in data:
        print("❌ Error: Falta la clave 'selections' en el JSON")
        return jsonify({'error': 'Falta la clave "selections"'}), 400

    try:
        selections = data['selections']
        print(f"✅ Selecciones recibidas: {selections}")
        processing_state['dip_selections'] = selections
        processing_state['is_processing'] = True
        threading.Thread(target=continuar_procesar_dips, daemon=True).start()
        return jsonify({'mensaje': 'Selecciones recibidas, continuando procesamiento'})
    except Exception as e:
        print(f"❌ Error al procesar selecciones: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)