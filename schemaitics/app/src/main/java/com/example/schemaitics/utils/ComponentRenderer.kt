package com.example.schemaitics.utils

import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.PointF
import android.graphics.RectF
import android.util.Log
import com.example.schemaitics.model.Component
import com.example.schemaitics.model.Pin
import com.example.schemaitics.SchematicView

object ComponentRenderer {

    private val componentPaint = Paint().apply {
        color = Color.BLACK
        style = Paint.Style.STROKE
        strokeWidth = 3f
        isAntiAlias = true
    }
//    private val componentPaintAmbiguo = Paint().apply {
//        color = Color.rgb(165, 42, 42) // Guindo cálido
//        style = Paint.Style.STROKE
//        strokeWidth = 3f
//        isAntiAlias = true
//    }

    private val pinPaint = Paint().apply {
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    private val connectionPaint = Paint().apply {
        color = Color.BLUE
        style = Paint.Style.STROKE
        strokeWidth = 3f
        isAntiAlias = true
    }

    private val textPaint = Paint().apply {
        color = Color.BLACK
        textSize = 24f
        isAntiAlias = true
    }

    private val nodePaint = Paint().apply {
        color = Color.BLACK
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    fun drawResistor(canvas: Canvas, x: Float, y: Float, pins: List<Pin>, rotation: Float, component: Component) {
        canvas.save()
        canvas.rotate(rotation, x, y)

        // Terminal izquierdo
        canvas.drawLine(x - 80f, y, x - 30f, y, componentPaint)
        // Terminal derecho
        canvas.drawLine(x + 30f, y, x + 80f, y, componentPaint)
        // Rectángulo
        canvas.drawRect(x - 30f, y - 10f, x + 30f, y + 10f, componentPaint)

        canvas.restore()
        drawPins(canvas, component, pins, x, y)
        drawLabel(canvas, component, x, y)
    }

    fun drawLed(canvas: Canvas, x: Float, y: Float, pins: List<Pin>, rotation: Float, component: Component) {
        canvas.save()
        canvas.rotate(rotation, x, y)
        Log.d("ComponentRenderer", "Dibujando LED en ($x, $y) con rotation $rotation usando componentPaintAmbiguo")

        val diodeSize = 20f
        // Terminal izquierdo (ánodo)
        canvas.drawLine(x - 80f, y, x - diodeSize, y, componentPaint)
        // Terminal derecho (cátodo)
        canvas.drawLine(x + diodeSize, y, x + 80f, y, componentPaint)

        // Triángulo
        val trianglePath = Path().apply {
            moveTo(x - diodeSize, y - diodeSize)
            lineTo(x - diodeSize, y + diodeSize)
            lineTo(x + diodeSize, y)
            close()
        }
        canvas.drawPath(trianglePath, componentPaint)

        // Línea cátodo
        canvas.drawLine(x + diodeSize, y - diodeSize, x + diodeSize, y + diodeSize, componentPaint)

        // Flechas
        canvas.drawLine(x - 5f, y - 25f, x + 10f, y - 40f, componentPaint.apply { strokeWidth = 2.5f })
        canvas.drawLine(x + 10f, y - 40f, x, y - 40f, componentPaint)
        canvas.drawLine(x + 10f, y - 40f, x + 10f, y - 30f, componentPaint)

        canvas.drawLine(x + 10f, y - 15f, x + 25f, y - 30f, componentPaint)
        canvas.drawLine(x + 25f, y - 30f, x + 15f, y - 30f, componentPaint)
        canvas.drawLine(x + 25f, y - 30f, x + 25f, y - 20f, componentPaint)

        canvas.restore()
        drawPins(canvas, component, pins, x, y)
        drawLabel(canvas, component, x, y)
    }

    fun drawButton(canvas: Canvas, x: Float, y: Float, pins: List<Pin>, rotation: Float, component: Component) {
        canvas.save()
        canvas.rotate(rotation, x, y)

        // Terminales horizontales
        canvas.drawLine(x - 80f, y, x - 25f, y, componentPaint)
        canvas.drawLine(x + 25f, y, x + 80f, y, componentPaint)

        // Círculos terminales internas
        canvas.drawCircle(x - 25f, y, 3f, componentPaint)
        canvas.drawCircle(x + 25f, y, 3f, componentPaint)

        // Línea superior (contacto móvil)
        canvas.drawLine(x - 20f, y - 30f, x + 20f, y - 30f, componentPaint)
        // Línea inferior (contacto fijo)
        canvas.drawLine(x - 30f, y - 20f, x + 30f, y - 20f, componentPaint)
        // Líneas verticales
        canvas.drawLine(x - 20f, y - 30f, x - 20f, y - 20f, componentPaint)
        canvas.drawLine(x + 20f, y - 30f, x + 20f, y - 20f, componentPaint)

        canvas.restore()
        drawPins(canvas, component, pins, x, y)
        drawLabel(canvas, component, x, y)
    }

    fun drawAndGate(canvas: Canvas, x: Float, y: Float, pins: List<Pin>, rotation: Float, component: Component) {
        canvas.save()
        canvas.rotate(rotation, x, y)

        val bodyWidth = 60f
        val bodyHeight = 80f
        val left = x - bodyWidth / 2
        val right = x + bodyWidth / 2
        val top = y - bodyHeight / 2
        val bottom = y + bodyHeight / 2

        // Rectángulo abierto (tres líneas)
        canvas.drawLine(left, top, right, top, componentPaint)        // Línea superior
        canvas.drawLine(left, top, left, bottom, componentPaint)      // Línea izquierda
        canvas.drawLine(left, bottom, right, bottom, componentPaint)  // Línea inferior

        // Semicírculo - equivalente a: left = centerX, right = right + bodyWidth / 2
        val arcRect = RectF(x, top, right + bodyWidth / 2, bottom)
        canvas.drawArc(arcRect, 270f, 180f, false, componentPaint)

        // Terminales de entrada
        canvas.drawLine(left - 30f, y - 20f, left, y - 20f, componentPaint)
        canvas.drawLine(left - 30f, y + 20f, left, y + 20f, componentPaint)

        // Terminal de salida
        canvas.drawLine(right + bodyWidth / 2, y, right + bodyWidth / 2 + 30f, y, componentPaint)

        canvas.restore()
        drawPins(canvas, component, pins, x, y)
        drawLabel(canvas, component, x, y)
    }

    fun drawPIC(canvas: Canvas, x: Float, y: Float, pins: List<Pin>, rotation: Float, component: Component) {
        canvas.save()
        canvas.rotate(rotation, x, y)

        // Rectángulo principal
        val bodyWidth = 240f
        val bodyHeight = 360f
        val left = x - bodyWidth / 2
        val right = x + bodyWidth / 2
        val top = y - bodyHeight / 2
        val bottom = y + bodyHeight / 2
        canvas.drawRect(left, top, right, bottom, componentPaint)

        canvas.restore()
        drawPins(canvas, component, pins, x, y)
        drawLabel(canvas, component, x, y)
    }

    fun drawNotGate(canvas: Canvas, x: Float, y: Float, pins: List<Pin>, rotation: Float, component: Component) {
        canvas.save()
        canvas.rotate(rotation, x, y)

        val triangleWidth = 60f
        val triangleHeight = 60f
        val circleRadius = 8f

        // Terminal entrada
        canvas.drawLine(x - 90f, y, x - triangleWidth / 2, y, componentPaint)

        // Triángulo
        val trianglePath = Path().apply {
            moveTo(x - triangleWidth / 2, y - triangleHeight / 2)
            lineTo(x - triangleWidth / 2, y + triangleHeight / 2)
            lineTo(x + triangleWidth / 2, y)
            close()
        }
        canvas.drawPath(trianglePath, componentPaint)

        // Círculo inversor
        canvas.drawCircle(x + triangleWidth / 2 + circleRadius, y, circleRadius, componentPaint)

        // Terminal salida
        canvas.drawLine(x + triangleWidth / 2 + 2 * circleRadius, y, x + 90f, y, componentPaint)

        canvas.restore()
        drawPins(canvas, component, pins, x, y)
        drawLabel(canvas, component, x, y)
    }

    fun drawOrGate(canvas: Canvas, x: Float, y: Float, pins: List<Pin>, rotation: Float, component: Component) {
        canvas.save()
        canvas.rotate(rotation, x, y)

        val bodyWidth = 80f
        val bodyHeight = 80f
        val left = x - bodyWidth / 2
        val right = x + bodyWidth / 2
        val top = y - bodyHeight / 2
        val bottom = y + bodyHeight / 2

        // Curva superior - equivalente a topLeft: Offset(left-90, top), size: Size(bodyWidth + 90f, bodyHeight)
        val arcRectTopBottom = RectF(left - 90f, top, left + bodyWidth, bottom)
        canvas.drawArc(arcRectTopBottom, -90f, 90f, false, componentPaint)

        // Curva inferior - mismo RectF que la superior pero diferente ángulo
        canvas.drawArc(arcRectTopBottom, 0f, 90f, false, componentPaint)

        // Curva central (paréntesis) - equivalente a topLeft: Offset(left-10 - bodyWidth/4, top), size: Size(bodyWidth/2, bodyHeight)
        val arcRectCenter = RectF(left - 10f - bodyWidth/4, top, left - 10f + bodyWidth/4, bottom)
        canvas.drawArc(arcRectCenter, -90f, 180f, false, componentPaint)

        // Terminales de entrada (A, B)
        canvas.drawLine(left - 30f, y - 20f, left + 15f, y - 20f, componentPaint)
        canvas.drawLine(left - 30f, y + 20f, left + 15f, y + 20f, componentPaint)

        // Terminal de salida (Y)
        canvas.drawLine(right - 5f, y, right + 20f, y, componentPaint)

        canvas.restore()
        drawPins(canvas, component, pins, x, y)
        drawLabel(canvas, component, x, y)
    }
    fun drawCapacitorCeramic(canvas: Canvas, x: Float, y: Float, pins: List<Pin>, rotation: Float, component: Component) {
        canvas.save()
        canvas.rotate(rotation, x, y)

        // Terminales
        canvas.drawLine(x - 80f, y, x - 10f, y, componentPaint) // Terminal izquierdo
        canvas.drawLine(x + 10f, y, x + 80f, y, componentPaint) // Terminal derecho

        // Placas del capacitor (dos líneas paralelas cortas)
        canvas.drawLine(x - 10f, y - 20f, x - 10f, y + 20f, componentPaint) // Placa izquierda
        canvas.drawLine(x + 10f, y - 20f, x + 10f, y + 20f, componentPaint) // Placa derecha

        canvas.restore()
        drawPins(canvas, component, pins, x, y)
        drawLabel(canvas, component, x, y)
    }

    fun drawCrystal(canvas: Canvas, x: Float, y: Float, pins: List<Pin>, rotation: Float, component: Component) {
        canvas.save()
        canvas.rotate(rotation, x, y)

        // Terminales
        canvas.drawLine(x - 80f, y, x - 30f, y, componentPaint) // Terminal izquierdo
        canvas.drawLine(x + 30f, y, x + 80f, y, componentPaint) // Terminal derecho

        // Rectángulo del cristal
        canvas.drawRect(x - 30f, y - 20f, x + 30f, y + 20f, componentPaint)

        // Líneas diagonales dentro del rectángulo
        canvas.drawLine(x - 20f, y - 10f, x + 20f, y + 10f, componentPaint)
        canvas.drawLine(x - 20f, y + 10f, x + 20f, y - 10f, componentPaint)

        canvas.restore()
        drawPins(canvas, component, pins, x, y)
        drawLabel(canvas, component, x, y)
    }
    fun drawUltrasonic(canvas: Canvas, x: Float, y: Float, pins: List<Pin>, rotation: Float, component: Component) {
        canvas.save()
        canvas.rotate(rotation, x, y)

        // Rectángulo principal (tamaño duplicado)
        val bodyWidth = 280f
        val bodyHeight = 140f
        val left = x - bodyWidth / 2
        val right = x + bodyWidth / 2
        val top = y - bodyHeight / 2
        val bottom = y + bodyHeight / 2
        canvas.drawRect(left, top, right, bottom, componentPaint)

        // Dos círculos dentro del rectángulo (tamaño duplicado)
        val circleRadius = 50f
        canvas.drawCircle(x - 70f, y, circleRadius, componentPaint) // Duplicado de -35f a -70f
        canvas.drawCircle(x + 70f, y, circleRadius, componentPaint) // Duplicado de +35f a +70f
        canvas.drawCircle(x - 70f, y, circleRadius - 10f, componentPaint) // Duplicado de -5f
        canvas.drawCircle(x + 70f, y, circleRadius - 10f, componentPaint) // Duplicado de -5f

        // Terminales en la parte inferior (asumimos 4 pines: VCC, Trig, Echo, GND) con offset duplicado
        val pinOffset = 40f
        canvas.drawLine(x - 60f, bottom, x - 60f, bottom + pinOffset, componentPaint) // Pin 1 (VCC), -30f → -60f
        canvas.drawLine(x - 20f, bottom, x - 20f, bottom + pinOffset, componentPaint) // Pin 2 (Trig), -10f → -20f
        canvas.drawLine(x + 20f, bottom, x + 20f, bottom + pinOffset, componentPaint) // Pin 3 (Echo), +10f → +20f
        canvas.drawLine(x + 60f, bottom, x + 60f, bottom + pinOffset, componentPaint) // Pin 4 (GND), +30f → +60f

        canvas.restore()
        drawPins(canvas, component, pins, x, y)
        drawLabel(canvas, component, x, y)
    }
    fun drawPower(canvas: Canvas, x: Float, y: Float, pins: List<Pin>, rotation: Float, component: Component) {
        canvas.save()
        canvas.rotate(rotation, x, y)

        // Rectángulo contenedor
        val rectHeight = pins.size * 40f + 20f // Duplicado el espaciado por pin (de 20f a 40f)
        canvas.drawRect(x - 20f, y - 10f, x + 20f, y + rectHeight - 10f, componentPaint)

        canvas.restore()
        drawPins(canvas, component, pins, x, y)
        drawLabel(canvas, component, x, y)
    }

    fun drawPlaceholder(canvas: Canvas, x: Float, y: Float, type: String, rotation: Float, component: Component) {
        canvas.save()
        canvas.rotate(rotation, x, y)
        canvas.drawRect(x - 80f, y - 40f, x + 80f, y + 40f, componentPaint)
        canvas.restore()
        drawLabel(canvas, component, x, y)
    }

    fun drawConnection(canvas: Canvas, start: PointF, end: PointF, pinName: String) {
        val paint = Paint(connectionPaint).apply {
            when (pinName.uppercase()) {
                "VCC" -> color = Color.GREEN
                "GND" -> color = Color.BLACK
                else -> color = Color.BLUE
            }
        }
        canvas.drawLine(start.x, start.y, end.x, end.y, paint)
        // Dibujar punto negro en el pin de inicio
        canvas.drawCircle(start.x, start.y, 5f, nodePaint)
    }

    fun drawNode(canvas: Canvas, node: PointF) {
        canvas.drawCircle(node.x, node.y, 5f, nodePaint)
    }

    private fun drawPins(canvas: Canvas, component: Component, pins: List<Pin>, componentX: Float, componentY: Float) {
        val pinTextPaint = Paint().apply {
            color = Color.BLACK
            textSize = 16f
            isAntiAlias = true
        }

        pins.forEach { pin ->
            val pinPos = SchematicView.getPinPositionForRenderer(component, pin, componentX, componentY)
            pinPaint.color = if (pin.connected) Color.GREEN else Color.RED
            canvas.drawCircle(pinPos.x, pinPos.y, 6f, pinPaint)
            canvas.save()
            canvas.rotate(component.rotation, pinPos.x, pinPos.y)
            canvas.drawText(pin.pin_name, pinPos.x + 10f, pinPos.y + 5f, pinTextPaint)
            canvas.restore()
            Log.d("ComponentRenderer", "Dibujando pin ${pin.pin_name} de ${component.type}_${component.instance} en (${pinPos.x}, ${pinPos.y})")
        }
    }

    private fun drawLabel(canvas: Canvas, component: Component, x: Float, y: Float) {
        val label = "${component.instance} ${component.type}"
        canvas.drawText(label, x - 40f, y - 50f, textPaint)
    }
}