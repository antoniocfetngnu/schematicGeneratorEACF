package com.example.schemaitics

import android.content.Context
import android.graphics.Canvas
import android.graphics.PointF
import android.graphics.RectF
import android.util.AttributeSet
import android.util.Log
import android.view.MotionEvent
import android.view.View
import com.example.schemaitics.model.Component
import com.example.schemaitics.model.Pin
import com.example.schemaitics.model.SchematicData
import com.example.schemaitics.utils.ComponentLayout
import com.example.schemaitics.utils.ComponentRenderer
import com.example.schemaitics.utils.NetUtils
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.PI
import kotlin.math.pow

class SchematicView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    var schematicData: SchematicData? = null
        set(value) {
            field = value
            positions = value?.let { ComponentLayout.calculateInitialPositions(it) } ?: mutableMapOf()
            Log.d("SchematicView", "SchematicData asignado: ${value?.components?.size} componentes")
            Log.d("SchematicView", "Posiciones: $positions")
            updateViewSize()
            invalidate()
        }

    private var positions: MutableMap<String, PointF> = mutableMapOf()
    private var selectedComponent: Component? = null
    private var lastTouch: PointF? = null
    private var touchDownTime: Long = 0
    var isCanvasMoveMode: Boolean = false

    init {
        setOnTouchListener { _, event ->
            handleTouch(event)
            true
        }
    }

    private fun updateViewSize() {
        if (positions.isEmpty()) {
            Log.d("SchematicView", "Posiciones vacías, lienzo no ajustado")
            return
        }
        val maxX = positions.values.maxOfOrNull { it.x } ?: 0f
        val maxY = positions.values.maxOfOrNull { it.y } ?: 0f
        minimumWidth = (maxX + 200f).toInt()
        minimumHeight = (maxY + 200f).toInt()
        Log.d("SchematicView", "Lienzo ajustado: width=$minimumWidth, height=$minimumHeight")
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        if (schematicData == null) {
            Log.d("SchematicView", "SchematicData es nulo, no se dibuja")
            return
        }

        Log.d("SchematicView", "Dibujando ${schematicData?.components?.size} componentes")
        drawConnections(canvas)
        schematicData?.components?.forEach { component ->
            val key = "${component.type}_${component.instance}"
            val position = positions[key] ?: PointF(100f, 100f)
            drawComponent(canvas, component, position.x, position.y)
        }
    }

    private fun drawComponent(canvas: Canvas, component: Component, x: Float, y: Float) {
        when (component.type) {
            "resistencia" -> ComponentRenderer.drawResistor(canvas, x, y, component.pins, component.rotation, component)
            "led" -> ComponentRenderer.drawLed(canvas, x, y, component.pins, component.rotation, component)
            "boton" -> ComponentRenderer.drawButton(canvas, x, y, component.pins, component.rotation, component)
            "7408_gate" -> ComponentRenderer.drawAndGate(canvas, x, y, component.pins, component.rotation, component)
            "7408_power" -> ComponentRenderer.drawPower(canvas, x, y, component.pins, component.rotation, component)
            "7404_gate" -> ComponentRenderer.drawNotGate(canvas, x, y, component.pins, component.rotation, component)
            "7404_power" -> ComponentRenderer.drawPower(canvas, x, y, component.pins, component.rotation, component)
            "7432_gate" -> ComponentRenderer.drawOrGate(canvas, x, y, component.pins, component.rotation, component)
            "7432_power" -> ComponentRenderer.drawPower(canvas, x, y, component.pins, component.rotation, component)
            "cap_cer" -> ComponentRenderer.drawCapacitorCeramic(canvas, x, y, component.pins, component.rotation, component)
            "cristal" -> ComponentRenderer.drawCrystal(canvas, x, y, component.pins, component.rotation, component)
            "usonic" -> ComponentRenderer.drawUltrasonic(canvas, x, y, component.pins, component.rotation, component)
            "18f2550" -> ComponentRenderer.drawPIC(canvas, x, y, component.pins, component.rotation, component)
            else -> ComponentRenderer.drawPlaceholder(canvas, x, y, component.type, component.rotation, component)
        }
    }

    private fun drawConnections(canvas: Canvas) {
        if (schematicData == null) return

        val netToPins = mutableMapOf<Int, MutableList<Pair<Component, Pin>>>()
        schematicData?.components?.forEach { component ->
            component.pins.forEach { pin ->
                if (pin.net != null && pin.connected) {
                    netToPins.getOrPut(pin.net!!) { mutableListOf() }.add(component to pin)
                }
            }
        }

        netToPins.forEach { (net, pins) ->
            if (pins.size < 2) return@forEach
            Log.d("SchematicView", "Dibujando conexiones para net $net con ${pins.size} pines")

            val pinPositions = pins.map { (component, pin) -> getPinPosition(component, pin) }

            when (pins.size) {
                2 -> {
                    // Conexión directa entre dos pines
                    val (comp1, pin1) = pins[0]
                    val (comp2, pin2) = pins[1]
                    val pos1 = pinPositions[0]
                    val pos2 = pinPositions[1]
                    ComponentRenderer.drawConnection(canvas, pos1, pos2, pin1.pin_name)
                }
                3 -> {
                    // Nodo central para 3 pines como estrella
                    val centerX = pinPositions.map { it.x }.average().toFloat()
                    val centerY = pinPositions.map { it.y }.average().toFloat()
                    val netNode = PointF(centerX, centerY)

                    pinPositions.forEachIndexed { idx, pinPos ->
                        ComponentRenderer.drawConnection(canvas, pinPos, netNode, pins[idx].second.pin_name)
                    }
                    ComponentRenderer.drawNode(canvas, netNode)
                }
                else -> {
                    // Para 4 o más pines, agrupar por cercanía y conectar dinámicamente
                    val intermediateNodes = mutableListOf<PointF>()
                    val remainingPins = pinPositions.toMutableList()

                    // Agrupar pines por cercanía hasta que no queden más
                    while (remainingPins.size >= 2) {
                        // Encontrar el par más cercano
                        var minDistance = Float.POSITIVE_INFINITY
                        var closestPair = Pair(0, 0)
                        for (i in 0 until remainingPins.size) {
                            for (j in i + 1 until remainingPins.size) {
                                val dist = distance(remainingPins[i], remainingPins[j])
                                if (dist < minDistance) {
                                    minDistance = dist
                                    closestPair = Pair(i, j)
                                }
                            }
                        }

                        // Calcular nodo intermedio para el par más cercano
                        val (i, j) = closestPair
                        val nodeX = (remainingPins[i].x + remainingPins[j].x) / 2
                        val nodeY = (remainingPins[i].y + remainingPins[j].y) / 2
                        val newNode = PointF(nodeX, nodeY)
                        intermediateNodes.add(newNode)

                        // Conectar los pines al nodo
                        ComponentRenderer.drawConnection(canvas, remainingPins[i], newNode, pins[pinPositions.indexOf(remainingPins[i])].second.pin_name)
                        ComponentRenderer.drawConnection(canvas, remainingPins[j], newNode, pins[pinPositions.indexOf(remainingPins[j])].second.pin_name)
                        ComponentRenderer.drawNode(canvas, newNode)

                        // Eliminar los pines procesados
                        if (j > i) {
                            remainingPins.removeAt(j)
                            remainingPins.removeAt(i)
                        } else {
                            remainingPins.removeAt(i)
                            remainingPins.removeAt(j)
                        }
                    }

                    // Conectar nodos intermedios entre sí si hay más de uno
                    if (intermediateNodes.size > 1) {
                        for (i in 0 until intermediateNodes.size - 1) {
                            ComponentRenderer.drawConnection(canvas, intermediateNodes[i], intermediateNodes[i + 1], "NODE")
                        }
                    }

                    // Manejar pines restantes (si es impar)
                    if (remainingPins.isNotEmpty()) {
                        val lastPinIdx = pinPositions.indexOf(remainingPins[0])
                        val lastNode = intermediateNodes.lastOrNull() ?: pinPositions.first()
                        ComponentRenderer.drawConnection(canvas, remainingPins[0], lastNode, pins[lastPinIdx].second.pin_name)
                    }
                }
            }
        }
    }

    // Función auxiliar para calcular distancia euclidiana
    private fun distance(p1: PointF, p2: PointF): Float {
        return kotlin.math.sqrt((p2.x - p1.x).pow(2) + (p2.y - p1.y).pow(2))
    }

    companion object {
        const val PIC_BODY_HEIGHT = 360f

        fun getPinPositionForRenderer(component: Component, pin: Pin, componentX: Float, componentY: Float): PointF {
            val angle = component.rotation * PI / 180
            Log.d("SchematicView", "Calculando pin para ${component.type}_${component.instance}, pin=${pin.pin_name}, rotation=${component.rotation}")

            val relativePos = when (component.type) {
                "resistencia" -> {
                    when (pin.pin_num) {
                        1 -> PointF(-80f, 0f)
                        2 -> PointF(80f, 0f)
                        else -> PointF(0f, 0f)
                    }
                }
                "led" -> {
                    when (pin.pin_num) {
                        1 -> PointF(80f, 0f)
                        2 -> PointF(-80f, 0f)
                        else -> PointF(0f, 0f)
                    }
                }
                "boton" -> {
                    when (pin.pin_num) {
                        1 -> PointF(-80f, 0f)
                        2 -> PointF(80f, 0f)
                        else -> PointF(0f, 0f)
                    }
                }
                "7408_gate" -> {
                    when {
                        pin.pin_name.uppercase().startsWith("A") -> PointF(-90f, -20f)
                        pin.pin_name.uppercase().startsWith("B") -> PointF(-90f, 20f)
                        pin.pin_name.uppercase().startsWith("Y") -> PointF(90f, 0f)
                        else -> PointF(0f, 0f)
                    }
                }
                "7404_gate" -> {
                    when {
                        pin.pin_name.uppercase().startsWith("A") -> PointF(-90f, 0f)
                        pin.pin_name.uppercase().startsWith("Y") -> PointF(90f, 0f)
                        else -> PointF(0f, 0f)
                    }
                }
                "7432_gate" -> {
                    when {
                        pin.pin_name.uppercase().startsWith("A") -> PointF(-90f, -20f)
                        pin.pin_name.uppercase().startsWith("B") -> PointF(-90f, 20f)
                        pin.pin_name.uppercase().startsWith("Y") -> PointF(90f, 0f)
                        else -> PointF(0f, 0f)
                    }
                }
                "7408_power", "7404_power", "7432_power" -> {
                    PointF(0f, component.pins.indexOf(pin) * 60f)
                }
                "cap_cer" -> {
                    when (pin.pin_num) {
                        1 -> PointF(-80f, 0f) // Terminal izquierdo
                        2 -> PointF(80f, 0f)  // Terminal derecho
                        else -> PointF(0f, 0f)
                    }
                }
                "cristal" -> {
                    when (pin.pin_num) {
                        1 -> PointF(-80f, 0f) // Terminal izquierdo
                        2 -> PointF(80f, 0f)  // Terminal derecho
                        else -> PointF(0f, 0f)
                    }
                }
                "usonic" -> {
                    when (pin.pin_num) {
                        1 -> PointF(-60f, 100f) // VCC, duplicado de -30f
                        2 -> PointF(-20f, 100f) // Trig, duplicado de -10f
                        3 -> PointF(20f, 100f)  // Echo, duplicado de +10f
                        4 -> PointF(60f, 100f)  // GND, duplicado de +30f
                        else -> PointF(0f, 0f)
                    }
                }
                "18f2550" -> {
                    when (pin.pin_num) {
                        in 1..14 -> {
                            val baseY = -180f + (pin.pin_num - 1) * (PIC_BODY_HEIGHT / 14)
                            when (pin.pin_num) {
                                8 -> PointF(-120f, -180f + 13 * (PIC_BODY_HEIGHT / 14)) // GND al final izquierdo
                                9 -> PointF(-120f, -180f + 7 * (PIC_BODY_HEIGHT / 14)) // OSC1 sube a posición 8
                                10 -> PointF(-120f, -180f + 8 * (PIC_BODY_HEIGHT / 14)) // OSC2 sube a posición 9
                                11 -> PointF(-120f, -180f + 9 * (PIC_BODY_HEIGHT / 14)) // RC0 sube a posición 10
                                12 -> PointF(-120f, -180f + 10 * (PIC_BODY_HEIGHT / 14)) // RC1 sube a posición 11
                                13 -> PointF(-120f, -180f + 11 * (PIC_BODY_HEIGHT / 14)) // RC2 sube a posición 12
                                14 -> PointF(-120f, -180f + 12 * (PIC_BODY_HEIGHT / 14)) // RC3 sube a posición 13
                                else -> PointF(-120f, baseY) // 1-7 en orden normal
                            }
                        }
                        in 15..28 -> {
                            val baseY = -180f + (pin.pin_num - 15) * (PIC_BODY_HEIGHT / 14)
                            when (pin.pin_num) {
                                19 -> PointF(120f, -180f + 13 * (PIC_BODY_HEIGHT / 14)) // GND al final derecho
                                20 -> PointF(120f, -180f + 0 * (PIC_BODY_HEIGHT / 14)) // VCC al inicio derecho
                                15 -> PointF(120f, -180f + 1 * (PIC_BODY_HEIGHT / 14)) // RC4 sube a posición 16
                                16 -> PointF(120f, -180f + 2 * (PIC_BODY_HEIGHT / 14)) // RC5 sube a posición 17
                                17 -> PointF(120f, -180f + 3 * (PIC_BODY_HEIGHT / 14)) // RC6 sube a posición 18
                                18 -> PointF(120f, -180f + 4 * (PIC_BODY_HEIGHT / 14)) // RC7 sube a posición 19
                                21 -> PointF(120f, -180f + 5 * (PIC_BODY_HEIGHT / 14)) // RB0 sube a posición 20
                                22 -> PointF(120f, -180f + 6 * (PIC_BODY_HEIGHT / 14)) // RB1 sube a posición 21
                                23 -> PointF(120f, -180f + 7 * (PIC_BODY_HEIGHT / 14)) // RB2 sube a posición 22
                                24 -> PointF(120f, -180f + 8 * (PIC_BODY_HEIGHT / 14)) // RB3 sube a posición 23
                                25 -> PointF(120f, -180f + 9 * (PIC_BODY_HEIGHT / 14)) // RB4 sube a posición 24
                                26 -> PointF(120f, -180f + 10 * (PIC_BODY_HEIGHT / 14)) // RB5 sube a posición 25
                                27 -> PointF(120f, -180f + 11 * (PIC_BODY_HEIGHT / 14)) // RB6 sube a posición 26
                                28 -> PointF(120f, -180f + 12 * (PIC_BODY_HEIGHT / 14)) // RB7 sube a posición 27
                                else -> PointF(120f, baseY) // Otros en orden normal
                            }
                        }
                        else -> PointF(0f, 0f)
                    }
                }
                else -> PointF(0f, 0f)
            }

            Log.d("SchematicView", "Pin ${pin.pin_name} relativa: (${relativePos.x}, ${relativePos.y})")
            val rotatedX = (relativePos.x * cos(angle) - relativePos.y * sin(angle)).toFloat()
            val rotatedY = (relativePos.x * sin(angle) + relativePos.y * cos(angle)).toFloat()
            Log.d("SchematicView", "Pin ${pin.pin_name} rotado: ($rotatedX, $rotatedY)")
            return PointF(componentX + rotatedX, componentY + rotatedY)
        }
    }

    private fun getPinPosition(component: Component, pin: Pin): PointF {
        val key = "${component.type}_${component.instance}"
        val basePos = positions[key] ?: PointF(100f, 100f)
        return getPinPositionForRenderer(component, pin, basePos.x, basePos.y)
    }

    private fun handleTouch(event: MotionEvent) {
        val touchX = event.x
        val touchY = event.y

        when (event.action) {
            MotionEvent.ACTION_DOWN -> {
                touchDownTime = System.currentTimeMillis()
                lastTouch = PointF(touchX, touchY)
                if (!isCanvasMoveMode) {
                    selectedComponent = findComponentAt(touchX, touchY)
                }
                Log.d("SchematicView", "Toque DOWN: CanvasMove=$isCanvasMoveMode, Componente=$selectedComponent")
            }
            MotionEvent.ACTION_MOVE -> {
                lastTouch?.let { last ->
                    val dx = touchX - last.x
                    val dy = touchY - last.y
                    if (isCanvasMoveMode) {
                        positions.forEach { (key, pos) ->
                            positions[key] = PointF(pos.x + dx, pos.y + dy)
                        }
                        Log.d("SchematicView", "Moviendo lienzo: dx=$dx, dy=$dy")
                    } else {
                        selectedComponent?.let { component ->
                            val key = "${component.type}_${component.instance}"
                            positions[key] = PointF(positions[key]?.x?.plus(dx) ?: 100f, positions[key]?.y?.plus(dy) ?: 100f)
                            Log.d("SchematicView", "Moviendo $key a (${positions[key]})")
                        }
                    }
                    lastTouch = PointF(touchX, touchY)
                    updateViewSize()
                    invalidate()
                }
            }
            MotionEvent.ACTION_UP -> {
                val touchDuration = System.currentTimeMillis() - touchDownTime
                if (touchDuration < 200 && selectedComponent != null && !isCanvasMoveMode) {
                    selectedComponent?.let { component ->
                        component.rotation = (component.rotation + 90f) % 360f
                        Log.d("SchematicView", "Rotando ${component.type} a ${component.rotation}°")
                        invalidate()
                    }
                    performClick()
                }
                selectedComponent = null
                lastTouch = null
                Log.d("SchematicView", "Toque UP")
            }
        }
    }

    override fun performClick(): Boolean {
        super.performClick()
        return true
    }

    private fun findComponentAt(x: Float, y: Float): Component? {
        schematicData?.components?.forEach { component ->
            val key = "${component.type}_${component.instance}"
            val pos = positions[key] ?: PointF(100f, 100f)
            val bounds = getComponentBounds(component, pos.x, pos.y)
            if (bounds.contains(x, y)) {
                Log.d("SchematicView", "Componente encontrado: ${component.type} en $bounds")
                return component
            }
        }
        return null
    }

    private fun getComponentBounds(component: Component, x: Float, y: Float): RectF {
        return when (component.type) {
            "resistencia" -> RectF(x - 100f, y - 30f, x + 100f, y + 30f)
            "led" -> RectF(x - 100f, y - 50f, x + 100f, y + 50f)
            "boton" -> RectF(x - 100f, y - 50f, x + 100f, y + 50f)
            "7408_gate", "7432_gate" -> RectF(x - 100f, y - 50f, x + 100f, y + 50f)
            "7404_gate" -> RectF(x - 100f, y - 50f, x + 100f, y + 50f)
            "7408_power", "7404_power", "7432_power" -> RectF(x - 20f, y - 20f, x + 20f, y + component.pins.size * 40f + 20f)
            "cap_cer" -> RectF(x - 100f, y - 40f, x + 100f, y + 40f)
            "cristal" -> RectF(x - 100f, y - 40f, x + 100f, y + 40f)
            "usonic" -> RectF(x - 280f, y - 140f, x + 280f, y + 180f)
            "18f2550" -> RectF(x - 120f, y - 180f, x + 120f, y + 180f) // Ajustado al tamaño del rectángulo
            else -> RectF(x - 100f, y - 50f, x + 100f, y + 50f)
        }
    }
}