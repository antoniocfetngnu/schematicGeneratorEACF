package com.example.schemaitics.utils

import android.graphics.PointF
import com.example.schemaitics.model.SchematicData

object ComponentLayout {
    fun calculateInitialPositions(schematic: SchematicData): MutableMap<String, PointF> {
        val positions = mutableMapOf<String, PointF>()
        val components = schematic.components
        val columns = 10 // Máximo 10 columnas
        val hSpacing = 20f // Espaciado horizontal
        val vSpacing = 20f // Espaciado vertical
        val powerSpacing = 60f // Espaciado horizontal para power
        val maxWidth = 160f // Ancho estimado por componente no-power
        val maxHeight = 100f // Alto estimado por componente no-power
        val powerWidth = 40f // Ancho de power
        val powerHeight = 60f // Alto de power
        val baseX = 50f // Desplazamiento inicial a la izquierda
        val baseY = 100f // Desplazamiento inicial arriba
        val powerExtraOffsetY = 400f // Offset adicional para power más abajo
        val canvasWidth = 1200f // Ancho estimado del lienzo para centrar power
        val powerXOffset = 300f

        // Separar componentes power y no-power
        val powerComponents = components.filter { it.type in listOf("7408_power", "7404_power", "7432_power") }
        val nonPowerComponents = components.filterNot { it.type in listOf("7408_power", "7404_power", "7432_power") }

        // Posicionar componentes no-power en cuadrícula
        nonPowerComponents.forEachIndexed { index, component ->
            val row = index / columns
            val col = index % columns
            val x = baseX + col * (maxWidth + hSpacing)
            val y = baseY + row * (maxHeight + vSpacing)
            val key = "${component.type}_${component.instance}"
            positions[key] = PointF(x, y)
        }

        // Calcular y máximo de no-power para posicionar power debajo
        val maxNonPowerY = positions.values.maxOfOrNull { it.y }?.plus(maxHeight) ?: 100f
        val powerY = maxNonPowerY + vSpacing + powerHeight + powerExtraOffsetY

        // Centrar componentes power horizontalmente
        val totalPowerWidth = powerComponents.size * powerWidth + (powerComponents.size - 1) * powerSpacing

        val powerStartX = (canvasWidth - totalPowerWidth) / 2 + powerXOffset // Centrar en el lienzo

        powerComponents.forEachIndexed { index, component ->
            val x = powerStartX + index * (powerWidth + powerSpacing)
            val y = powerY
            val key = "${component.type}_${component.instance}"
            positions[key] = PointF(x, y)
        }

        return positions
    }
}