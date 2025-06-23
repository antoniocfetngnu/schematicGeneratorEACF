package com.example.schemaitics

import android.content.Intent
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import okhttp3.*
import org.json.JSONObject
import java.io.File
import java.io.IOException
import android.util.Base64
import android.util.Log
import org.json.JSONException


class ProcessingActivity : AppCompatActivity() {
    private val client = OkHttpClient()
    private var isPolling = false
    private val REQUEST_DIP_SELECTION = 100
    private val handler = Handler(Looper.getMainLooper()) // Handler para manejar retrasos

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_processing)
        startPolling()
    }

    private fun startPolling() {
        if (isPolling) return
        isPolling = true

        val request = Request.Builder()
            .url("${Constants.SERVER_IP}/estado-detecciones")
            .build()

//        pollStatus(request)
        handler.postDelayed({
            if (isPolling) {
                pollStatus(request)
            }
        }, 5000) // 3000 ms = 3 segundos
    }

    private fun pollStatus(request: Request) {
        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                runOnUiThread {
                    Log.e("ProcessingActivity", "Error en polling: ${e.message}")
                    Toast.makeText(this@ProcessingActivity, "Error: ${e.message}", Toast.LENGTH_SHORT).show()
                    isPolling = false
                    setResult(RESULT_CANCELED)
                    finish()
                }
            }

            override fun onResponse(call: Call, response: Response) {
                if (!response.isSuccessful) {
                    runOnUiThread {
                        Log.e("ProcessingActivity", "Error en el servidor: ${response.code}")
                        Toast.makeText(this@ProcessingActivity, "Error en el servidor: ${response.code}", Toast.LENGTH_SHORT).show()
                        isPolling = false
                        setResult(RESULT_CANCELED)
                        finish()
                    }
                    return
                }

                val responseBody = response.body?.string()
                Log.d("ProcessingActivity", "Respuesta de /estado-detecciones: $responseBody")
                val jsonResponse = JSONObject(responseBody ?: "{}")
                val mensaje = jsonResponse.optString("mensaje", "Respuesta inválida")

                runOnUiThread {
                    when (mensaje) {
                        "Detecciones en curso" -> {
                            Log.d("ProcessingActivity", "Procesando detecciones...")
                            Toast.makeText(this@ProcessingActivity, "Procesando...", Toast.LENGTH_SHORT).show()
                            if (isPolling) {
//                                pollStatus(request)
                                // Programar el próximo polling con 3 segundos de retraso
                                handler.postDelayed({
                                    if (isPolling) {
                                        pollStatus(request)
                                    }
                                }, 3000) // 3000 ms = 3 segundos
                            }
                        }
                        "DIPs detectados" -> {
                            Log.d("ProcessingActivity", "DIPs detectados, iniciando DipSelectionActivity")
                            val dipDetections = jsonResponse.getJSONArray("dip_detections")
                            Log.d("ProcessingActivity", "dip_detections: $dipDetections")
                            val imageBase64 = jsonResponse.getString("dip_image_base64")
                            val imageFile = saveImageToFile(imageBase64)
                            if (imageFile == null) {
                                Log.e("ProcessingActivity", "Error al guardar imagen")
                                Toast.makeText(this@ProcessingActivity, "Error al guardar la imagen", Toast.LENGTH_SHORT).show()
                                isPolling = false
                                setResult(RESULT_CANCELED)
                                finish()
                                return@runOnUiThread
                            }
                            val intent = Intent(this@ProcessingActivity, DipSelectionActivity::class.java).apply {
                                putExtra("dip_image_path", imageFile.absolutePath)
                                putExtra("dip_detections", dipDetections.toString())
                                putExtra("dip_netlist", jsonResponse.getJSONObject("dip_netlist").toString())
                            }
                            startActivityForResult(intent, REQUEST_DIP_SELECTION)
                            isPolling = false
                        }
                        "Detecciones completadas con éxito" -> {
                            Log.d("ProcessingActivity", "Detecciones completadas")
                            Toast.makeText(this@ProcessingActivity, "Procesamiento completado", Toast.LENGTH_LONG).show()
                            isPolling = false
                            // Extraer schematic_data del JSON de forma segura
                            val schematicData = try {
                                if (jsonResponse.has("schematic_data") && !jsonResponse.isNull("schematic_data")) {
                                    jsonResponse.getJSONArray("schematic_data").toString()
                                } else {
                                    Log.e("ProcessingActivity", "schematic_data es null o no existe")
                                    "No schematic data received"
                                }
                            } catch (e: JSONException) {
                                Log.e("ProcessingActivity", "Error al parsear schematic_data: ${e.message}")
                                "Error parsing schematic data"
                            }
                            // Lanzar la nueva actividad
                            val intent = Intent(this@ProcessingActivity, SchematicViewerActivity::class.java).apply {
                                putExtra("schematic_data", schematicData)
                            }
                            startActivity(intent)
                            setResult(RESULT_OK)
                            finish()
                        }
                        else -> {
                            Log.e("ProcessingActivity", "Mensaje desconocido: $mensaje")
                            Toast.makeText(this@ProcessingActivity, "Error: $mensaje", Toast.LENGTH_SHORT).show()
                            isPolling = false
                            setResult(RESULT_CANCELED)
                            finish()
                        }
                    }
                }
            }
        })
    }

    private fun saveImageToFile(base64: String): File? {
        return try {
            val imageBytes = Base64.decode(base64, Base64.DEFAULT)
            val file = File(cacheDir, "dip_image_${System.currentTimeMillis()}.jpg")
            file.writeBytes(imageBytes)
            file
        } catch (e: Exception) {
            Log.e("ProcessingActivity", "Error al guardar imagen: ${e.message}")
            null
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        Log.d("ProcessingActivity", "onActivityResult: requestCode=$requestCode, resultCode=$resultCode")
        if (requestCode == REQUEST_DIP_SELECTION) {
            if (resultCode == RESULT_OK) {
                Log.d("ProcessingActivity", "Selecciones de DIPs recibidas, reanudando polling")
                Toast.makeText(this, "Selecciones de DIPs enviadas, continuando procesamiento", Toast.LENGTH_SHORT).show()
                startPolling() // Reanudar polling para esperar detecciones completadas
            } else {
                Log.e("ProcessingActivity", "Selección de DIPs cancelada")
                Toast.makeText(this, "Selección de DIPs cancelada", Toast.LENGTH_SHORT).show()
                setResult(RESULT_CANCELED)
                finish()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        isPolling = false
        handler.removeCallbacksAndMessages(null) // Limpiar cualquier callback pendiente
    }
}