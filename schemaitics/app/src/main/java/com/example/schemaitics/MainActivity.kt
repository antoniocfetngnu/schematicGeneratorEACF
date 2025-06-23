package com.example.schemaitics

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.os.Bundle
import android.os.Build
import android.util.Base64
import android.util.Log
import android.view.View
import android.view.ViewTreeObserver
import android.widget.Button
import android.widget.Toast
import android.widget.ToggleButton
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import java.io.ByteArrayOutputStream
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import android.view.WindowInsets
import android.view.WindowInsetsController
import android.view.WindowManager
import com.example.schemaitics.R
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.io.IOException

class MainActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var captureButton: Button
    private var imageCapture: ImageCapture? = null
    private var flashMode = ImageCapture.FLASH_MODE_OFF
    private var cameraControl: CameraControl? = null
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var imageAnalyzer: ImageAnalysis
    private var leftBorderView: View? = null
    private var rightBorderView: View? = null
    private var protoboardDetected = false
    private val client = OkHttpClient()

    private val cannyThreshold1 = 70.0
    private val cannyThreshold2 = 150.0
    private val minEdgePercentage = 0.30

    private var isPhotoBlocked = false


    companion object {
        private const val TAG = "CameraXOpenCV"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
        private const val SERVER_URL = "${Constants.SERVER_IP}/procesar-imagen"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        if (!OpenCVLoader.initDebug()) {
            Log.e(TAG, "No se pudo inicializar OpenCV")
        } else {
            Log.d(TAG, "OpenCV inicializado correctamente")
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            window.attributes.layoutInDisplayCutoutMode =
                WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_SHORT_EDGES
        }
        window.setFlags(
            WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS,
            WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS
        )
        setContentView(R.layout.activity_main)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            window.insetsController?.hide(WindowInsets.Type.statusBars() or WindowInsets.Type.navigationBars())
            window.insetsController?.systemBarsBehavior = WindowInsetsController.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
        } else {
            @Suppress("DEPRECATION")
            window.decorView.systemUiVisibility = (
                    View.SYSTEM_UI_FLAG_FULLSCREEN
                            or View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                            or View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
                            or View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
                            or View.SYSTEM_UI_FLAG_LAYOUT_STABLE
                            or View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
                    )
        }

        previewView = findViewById(R.id.previewView)
        val overlayView: View = findViewById(R.id.overlayView)
        captureButton = findViewById(R.id.captureButton)
        leftBorderView = findViewById(R.id.leftBorderView)
        rightBorderView = findViewById(R.id.rightBorderView)

        captureButton.isEnabled = false

        previewView.viewTreeObserver.addOnGlobalLayoutListener(object : ViewTreeObserver.OnGlobalLayoutListener {
            override fun onGlobalLayout() {
                previewView.viewTreeObserver.removeOnGlobalLayoutListener(this)
                val previewWidth = previewView.width
                val previewHeight = previewView.height
                val overlayWidth = (previewWidth * 0.9).toInt()
                val overlayHeight = (previewHeight * 0.7).toInt()
                overlayView.layoutParams.width = overlayWidth
                overlayView.layoutParams.height = overlayHeight
                overlayView.requestLayout()
            }
        })

        cameraExecutor = Executors.newSingleThreadExecutor()

        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }

        captureButton.setOnClickListener { takePhoto() }

        val torchButton: ToggleButton = findViewById(R.id.torchButton)
        torchButton.setOnCheckedChangeListener { _, isChecked ->
            cameraControl?.enableTorch(isChecked)
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        imageCapture = ImageCapture.Builder()
            .setFlashMode(flashMode)
            .build()

        imageAnalyzer = ImageAnalysis.Builder()
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
            .also {
                it.setAnalyzer(cameraExecutor, ImageAnalyzer { edgesDetected ->
                    protoboardDetected = edgesDetected
                    runOnUiThread {
                        captureButton.isEnabled = edgesDetected
                        if (edgesDetected) {
                            captureButton.setBackgroundColor(ContextCompat.getColor(this, android.R.color.holo_green_light))
                        } else {
                            captureButton.setBackgroundColor(ContextCompat.getColor(this, android.R.color.darker_gray))
                        }
                    }
                })
            }

        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }

            imageCapture = ImageCapture.Builder()
                .setFlashMode(flashMode)
                .build()

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                val camera = cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageCapture, imageAnalyzer)
                cameraControl = camera.cameraControl
            } catch (exc: Exception) {
                Log.e(TAG, "Error al iniciar la cámara: ${exc.message}", exc)
                Toast.makeText(this, "Error al iniciar la cámara: ${exc.message}", Toast.LENGTH_SHORT).show()
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun takePhoto() {
        if (!protoboardDetected || isPhotoBlocked) {
            if (isPhotoBlocked) {
                Toast.makeText(this, "Espera 3 segundos antes de tomar otra foto", Toast.LENGTH_SHORT).show()
            } else {
                Toast.makeText(this, "Alinea el protoboard correctamente entre las líneas verdes", Toast.LENGTH_SHORT).show()
            }
            return
        }

        val imageCapture = imageCapture?.apply { flashMode = this@MainActivity.flashMode } ?: return

        captureButton.isEnabled = false
        isPhotoBlocked = true

        imageCapture.takePicture(
            cameraExecutor,
            object : ImageCapture.OnImageCapturedCallback() {
                override fun onCaptureSuccess(image: ImageProxy) {
                    runOnUiThread {
                        Toast.makeText(this@MainActivity, "Foto capturada correctamente, esperando respuesta...", Toast.LENGTH_SHORT).show()
                    }
                    try {
                        val bitmap = imageProxyToBitmap(image)
                        if (bitmap != null) {
                            val croppedBitmap = cropCapturedImage(bitmap)
                            if (croppedBitmap != null) {
                                sendImageToServer(croppedBitmap)

                                // Iniciar el timer de 3 segundos DESPUÉS de enviar al servidor
                                startPhotoBlockTimer()

                                croppedBitmap.recycle()
                            } else {
                                runOnUiThread {
                                    Toast.makeText(this@MainActivity, "Error al recortar la imagen", Toast.LENGTH_SHORT).show()
                                    isPhotoBlocked = false // Desbloquear en caso de error
                                    // No habilitar el botón aquí, dejar que el ImageAnalyzer lo maneje
                                }
                            }
                        } else {
                            runOnUiThread {
                                Toast.makeText(this@MainActivity, "Error al convertir la imagen", Toast.LENGTH_SHORT).show()
                                isPhotoBlocked = false // Desbloquear en caso de error
                            }
                        }
                    } catch (e: Exception) {
                        Log.e(TAG, "Error al procesar la imagen: ${e.message}", e)
                        runOnUiThread {
                            Toast.makeText(this@MainActivity, "Error al procesar la imagen: ${e.message}", Toast.LENGTH_SHORT).show()
                            isPhotoBlocked = false // Desbloquear en caso de error
                        }
                    } finally {
                        image.close()
                    }
                }

                override fun onError(exception: ImageCaptureException) {
                    runOnUiThread {
                        Toast.makeText(this@MainActivity, "Error al tomar foto: ${exception.message}", Toast.LENGTH_SHORT).show()
                        isPhotoBlocked = false // Desbloquear en caso de error
                    }
                }
            }
        )
    }

    // Agregar este nuevo método
    private fun startPhotoBlockTimer() {
        // Usar Handler para desbloquear después de 3 segundos
        android.os.Handler(android.os.Looper.getMainLooper()).postDelayed({
            isPhotoBlocked = false
            // El ImageAnalyzer se encargará de habilitar el botón si las condiciones se cumplen
        }, 3000) // 3 segundos
    }

    private fun cropCapturedImage(bitmap: Bitmap): Bitmap? {
        try {
            val previewWidth = previewView.width
            val previewHeight = previewView.height
            val imageWidth = bitmap.width
            val imageHeight = bitmap.height

            val previewRatio = previewWidth.toFloat() / previewHeight
            val imageRatio = imageWidth.toFloat() / imageHeight

            val cropWidth: Int
            val cropHeight: Int
            val offsetX: Int
            val offsetY: Int

            if (imageRatio > previewRatio) {
                cropHeight = imageHeight
                cropWidth = (imageHeight * previewRatio).toInt()
                offsetX = (imageWidth - cropWidth) / 2
                offsetY = 0
            } else {
                cropWidth = imageWidth
                cropHeight = (imageWidth / previewRatio).toInt()
                offsetX = 0
                offsetY = (imageHeight - cropHeight) / 2
            }

            return Bitmap.createBitmap(bitmap, offsetX, offsetY, cropWidth, cropHeight)
        } catch (e: Exception) {
            Log.e(TAG, "Error al recortar la imagen: ${e.message}", e)
            return null
        }
    }

    private fun sendImageToServer(bitmap: Bitmap) {
        val outputStream = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, 95, outputStream)
        val imageBytes = outputStream.toByteArray()
        val base64Image = Base64.encodeToString(imageBytes, Base64.DEFAULT)

        val json = JSONObject().apply {
            put("imagen_base64", base64Image)
        }
        val requestBody = json.toString().toRequestBody("application/json".toMediaType())

        val request = Request.Builder()
            .url(SERVER_URL)
            .post(requestBody)
            .build()

        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                runOnUiThread {
                    Toast.makeText(this@MainActivity, "Error al enviar la imagen: ${e.message}", Toast.LENGTH_SHORT).show()
                    captureButton.isEnabled = true
                }
                Log.e(TAG, "Error en la solicitud: ${e.message}", e)
            }

            override fun onResponse(call: Call, response: Response) {
                val responseBody = response.body?.string()
                runOnUiThread {
                    if (response.isSuccessful && responseBody != null) {
                        try {
                            val jsonResponse = JSONObject(responseBody)
                            val message = jsonResponse.getString("mensaje")
                            Toast.makeText(this@MainActivity, message, Toast.LENGTH_SHORT).show()
                            val intent = Intent(this@MainActivity, ProcessingActivity::class.java)
                            startActivity(intent)
                        } catch (e: Exception) {
                            Toast.makeText(this@MainActivity, "Error al procesar la respuesta del servidor", Toast.LENGTH_SHORT).show()
                            captureButton.isEnabled = true
                        }
                    } else {
                        Toast.makeText(this@MainActivity, "Error en el servidor: ${response.message}", Toast.LENGTH_SHORT).show()
                        captureButton.isEnabled = true
                    }
                }
            }
        })
    }

    private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap? {
        try {
            if (imageProxy.format == ImageFormat.YUV_420_888) {
                return yuv420888ToBitmap(imageProxy)
            } else {
                val buffer = imageProxy.planes[0].buffer
                val bytes = ByteArray(buffer.remaining())
                buffer.get(bytes)
                val bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size) ?: return null
                val matrix = Matrix()
                matrix.postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())
                return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error en imageProxyToBitmap: ${e.message}", e)
            return null
        }
    }

    private fun yuv420888ToBitmap(image: ImageProxy): Bitmap? {
        val planes = image.planes
        val yBuffer = planes[0].buffer
        val uBuffer = planes[1].buffer
        val vBuffer = planes[2].buffer
        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()
        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), 100, out)
        val imageBytes = out.toByteArray()
        return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
    }

    private fun allPermissionsGranted() =
        ContextCompat.checkSelfPermission(baseContext, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                startCamera()
            } else {
                Toast.makeText(this, "Permiso de cámara no concedido.", Toast.LENGTH_SHORT).show()
                finish()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }

    private inner class ImageAnalyzer(private val listener: (Boolean) -> Unit) : ImageAnalysis.Analyzer {
        private var frameCounter = 0
        private val PROCESS_EVERY_N_FRAMES = 3

        override fun analyze(image: ImageProxy) {
            frameCounter++
            if (frameCounter % PROCESS_EVERY_N_FRAMES == 0) {
                val bitmap = imageProxyToBitmap(image)
                if (bitmap != null) {
                    val leftBorderPosition = getViewPositionInPreview(leftBorderView)
                    val rightBorderPosition = getViewPositionInPreview(rightBorderView)
                    if (leftBorderPosition != null && rightBorderPosition != null) {
                        val edgesDetected = detectEdgesAtBorders(bitmap, leftBorderPosition, rightBorderPosition)
                        // Solo habilitar el botón si los bordes son detectados Y no estamos bloqueados
                        listener(edgesDetected && !isPhotoBlocked)
                    } else {
                        listener(false)
                        Log.d(TAG, "No se pudieron obtener las posiciones de las líneas verdes")
                    }
                } else {
                    Log.e(TAG, "Error al convertir ImageProxy a Bitmap")
                    listener(false)
                }
            }
            image.close()
        }


        private fun getViewPositionInPreview(view: View?): Point? {
            if (view == null) return null
            val location = IntArray(2)
            view.getLocationInWindow(location)
            val previewLocation = IntArray(2)
            previewView.getLocationInWindow(previewLocation)
            val x = (location[0] - previewLocation[0]).toDouble()
            val y = (location[1] - previewLocation[1]).toDouble()
            Log.d(TAG, "Posición de la línea verde: x=$x, y=$y")
            return Point(x, y)
        }

        private fun detectEdgesAtBorders(bitmap: Bitmap, leftBorderPosition: Point, rightBorderPosition: Point): Boolean {
            val mat = Mat()
            Utils.bitmapToMat(bitmap, mat)
            val grayMat = Mat()
            Imgproc.cvtColor(mat, grayMat, Imgproc.COLOR_BGR2GRAY)
            val edgesMat = Mat()
            Imgproc.Canny(grayMat, edgesMat, cannyThreshold1, cannyThreshold2)
            val scaleX = mat.width().toFloat() / previewView.width
            val scaleY = mat.height().toFloat() / previewView.height
            val leftBorderX = (leftBorderPosition.x * scaleX).toInt()
            val rightBorderX = (rightBorderPosition.x * scaleX).toInt()
            val leftEdgeCount = countEdgesInVerticalLine(edgesMat, leftBorderX)
            val rightEdgeCount = countEdgesInVerticalLine(edgesMat, rightBorderX)
            val totalHeight = edgesMat.height()
            val minEdgeCount = (totalHeight * minEdgePercentage).toInt()
            val leftPercentage = (leftEdgeCount * 100.0 / totalHeight).toInt()
            val rightPercentage = (rightEdgeCount * 100.0 / totalHeight).toInt()
            val minPercentage = (minEdgePercentage * 100).toInt()
            Log.d(TAG, "Detección: [Izq: $leftEdgeCount/$totalHeight ($leftPercentage%)] [Der: $rightEdgeCount/$totalHeight ($rightPercentage%)] [Min requerido: $minPercentage%]")
            Log.d(TAG, "Resultado: ${if (leftEdgeCount >= minEdgeCount && rightEdgeCount >= minEdgeCount) "DETECTADO ✓" else "NO DETECTADO ✗"}")
            mat.release()
            grayMat.release()
            edgesMat.release()
            return leftEdgeCount >= minEdgeCount && rightEdgeCount >= minEdgeCount
        }

        private fun countEdgesInVerticalLine(edgesMat: Mat, x: Int): Int {
            val detectionWidth = 2
            var count = 0
            for (i in -detectionWidth until detectionWidth + 1) {
                val currentX = x + i
                if (currentX < 0 || currentX >= edgesMat.width()) continue
                for (y in 0 until edgesMat.height()) {
                    val pixel = edgesMat.get(y, currentX)
                    if (pixel != null && pixel[0] > 0) {
                        count++
                    }
                }
            }
            return count
        }
    }
}