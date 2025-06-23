package com.example.schemaitics

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.drawable.ColorDrawable
import android.os.Bundle
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.IOException
import android.util.Log
import androidx.core.graphics.scale

class DipSelectionActivity : AppCompatActivity() {
    private lateinit var imageDips: ImageView
    private lateinit var dipSelectionContainer: LinearLayout
    private lateinit var buttonSendSelections: Button
    private val selections = mutableMapOf<Int, String>()
    private val client = OkHttpClient()
    private val colors = listOf(
        Color.rgb(255, 165, 0), // Naranja
        Color.rgb(0, 255, 0),   // Verde
        Color.rgb(0, 0, 255),   // Azul
        Color.rgb(255, 255, 0), // Amarillo
        Color.rgb(255, 0, 255)  // Magenta
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_dip_selection)

        imageDips = findViewById(R.id.image_dips)
        dipSelectionContainer = findViewById(R.id.dip_selection_container)
        buttonSendSelections = findViewById(R.id.button_send_selections)

        val dipImagePath = intent.getStringExtra("dip_image_path")
        val dipDetectionsJson = intent.getStringExtra("dip_detections")
        val dipNetlistJson = intent.getStringExtra("dip_netlist")

        // Cargar y mostrar imagen desde archivo
        try {
            if (dipImagePath != null) {
                val file = File(dipImagePath)
                if (!file.exists()) {
                    throw IllegalStateException("El archivo de imagen no existe")
                }
//                val bitmap = BitmapFactory.decodeFile(dipImagePath) ?: throw IllegalStateException("No se pudo decodificar la imagen")
//                imageDips.setImageBitmap(bitmap)
                val originalBitmap = BitmapFactory.decodeFile(dipImagePath)
                val distortedBitmap = originalBitmap.scale(
                    originalBitmap.width,
                    (originalBitmap.height * 0.5f).toInt()
                )
                imageDips.setImageBitmap(distortedBitmap)
                // Opcional: eliminar el archivo después de usarlo
                file.delete()
            } else {
                throw IllegalArgumentException("dip_image_path es nulo")
            }
        } catch (e: Exception) {
            Log.e("DipSelectionActivity", "Error al cargar imagen: ${e.message}")
            Toast.makeText(this, "Error al cargar imagen: ${e.message}", Toast.LENGTH_LONG).show()
            setResult(RESULT_CANCELED)
            finish()
            return
        }

        // Parsear datos
        try {
            val dipDetections = JSONArray(dipDetectionsJson)
            Log.d("DipSelectionActivity", "dipDetections: $dipDetections")
            val dipNetlist = JSONObject(dipNetlistJson)
            val dipTypes = dipNetlist.keys().asSequence().toList().sorted()
            // Añadir texto de instrucción
            val instructionText = TextView(this).apply {
                text = "Elija el DIP correspondiente"
                textSize = 18f
                setTextColor(Color.BLACK)
                setTypeface(null, android.graphics.Typeface.BOLD)
                layoutParams = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.WRAP_CONTENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT
                ).apply { setMargins(0, 16, 0, 16) }
            }
            dipSelectionContainer.addView(instructionText)
            // Crear un Spinner por cada DIP
            for (i in 0 until dipDetections.length()) {
                val dip = dipDetections.getJSONObject(i)
                val dipId = dip.getInt("id")
                Log.d("DipSelectionActivity", "DIP $dipId: color=${dip.optJSONArray("color")}")


                val layout = LinearLayout(this).apply {
                    orientation = LinearLayout.HORIZONTAL
                    layoutParams = LinearLayout.LayoutParams(
                        LinearLayout.LayoutParams.MATCH_PARENT,
                        LinearLayout.LayoutParams.WRAP_CONTENT
                    ).apply { setMargins(0, 16, 0, 16) }
                }

                // Indicador de color
                val colorView = View(this).apply {
                    layoutParams = LinearLayout.LayoutParams(50, 50).apply { setMargins(0, 0, 16, 0) }
                    try {
                        val colorArray = dip.getJSONArray("color")
                        val color = Color.rgb(
                            colorArray.getInt(0),
                            colorArray.getInt(1),
                            colorArray.getInt(2)
                        )
                        background = ColorDrawable(color)
                    } catch (e: Exception) {
                        Log.w("DipSelectionActivity", "Color no encontrado para DIP $dipId, usando color por defecto: ${e.message}")
                        background = ColorDrawable(colors[i % colors.size])
                    }
                }

                // Spinner
                val spinner = Spinner(this).apply {
                    layoutParams = LinearLayout.LayoutParams(
                        0,
                        LinearLayout.LayoutParams.WRAP_CONTENT,
                        1f
                    )
                }

                val adapter = ArrayAdapter(
                    this@DipSelectionActivity,
                    android.R.layout.simple_spinner_item,
                    listOf("Seleccionar...") + dipTypes
                )
                adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
                spinner.adapter = adapter

                spinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
                    override fun onItemSelected(parent: AdapterView<*>, view: View?, position: Int, id: Long) {
                        if (position > 0) {
                            selections[dipId] = dipTypes[position - 1]
                        } else {
                            selections.remove(dipId)
                        }
                        buttonSendSelections.isEnabled = selections.size == dipDetections.length()
                    }

                    override fun onNothingSelected(parent: AdapterView<*>) {
                        selections.remove(dipId)
                        buttonSendSelections.isEnabled = false
                    }
                }

                layout.addView(colorView)
                layout.addView(spinner)
                dipSelectionContainer.addView(layout)
            }
        } catch (e: Exception) {
            Log.e("DipSelectionActivity", "Error al procesar datos: ${e.message}")
            Toast.makeText(this, "Error al procesar datos: ${e.message}", Toast.LENGTH_LONG).show()
            setResult(RESULT_CANCELED)
            finish()
            return
        }

        // Enviar selecciones
        buttonSendSelections.setOnClickListener {
            Log.d("DipSelectionActivity", "Tamaño de selections: ${selections.size}")
            val selectionsArray = JSONArray()
            selections.forEach { (id, type) ->
                selectionsArray.put(JSONObject().apply {
                    put("id", id)
                    put("type", type)
                })
            }

            val requestBody = JSONObject().apply {
                put("selections", selectionsArray)
            }
            Log.d("DipSelectionActivity", "Enviando selecciones: ${requestBody.toString()}")
            val request = Request.Builder()
                .url("${Constants.SERVER_IP}/dip-selections")
                .post(RequestBody.create("application/json".toMediaType(), requestBody.toString()))
                .build()

            client.newCall(request).enqueue(object : Callback {
                override fun onFailure(call: Call, e: IOException) {
                    runOnUiThread {
                        Log.e("DipSelectionActivity", "Error en la solicitud: ${e.message}")
                        if (!isFinishing) {
                            Toast.makeText(this@DipSelectionActivity, "Error: ${e.message}", Toast.LENGTH_SHORT).show()
                            setResult(RESULT_CANCELED)
                            finish()
                        }
                    }
                }

                override fun onResponse(call: Call, response: Response) {
                    val responseBody = try {
                        response.body?.string()
                    } catch (e: Exception) {
                        Log.e("DipSelectionActivity", "Error al leer cuerpo de respuesta: ${e.message}", e)
                        null
                    }

                    runOnUiThread {
                        try {
                            if (response.isSuccessful && responseBody != null) {
                                Log.d("DipSelectionActivity", "Respuesta del servidor: $responseBody")
                                val jsonResponse = JSONObject(responseBody)
                                val mensaje = jsonResponse.optString("mensaje", jsonResponse.optString("error", "Sin mensaje"))
                                Log.d("DipSelectionActivity", "Mensaje del servidor: $mensaje")
                                if (!isFinishing) {
                                    Toast.makeText(this@DipSelectionActivity, mensaje, Toast.LENGTH_SHORT).show()
                                    val resultIntent = Intent()
                                    resultIntent.putExtra("selections", requestBody.toString())
                                    setResult(RESULT_OK, resultIntent)
                                    finish()
                                }
                            } else {
                                val errorMsg = if (responseBody == null) {
                                    "Cuerpo de respuesta vacío"
                                } else {
                                    "Error en el servidor: ${response.code}"
                                }
                                Log.e("DipSelectionActivity", errorMsg)
                                if (!isFinishing) {
                                    Toast.makeText(this@DipSelectionActivity, errorMsg, Toast.LENGTH_SHORT).show()
                                    setResult(RESULT_CANCELED)
                                    finish()
                                }
                            }
                        } catch (e: Exception) {
                            Log.e("DipSelectionActivity", "Error al procesar respuesta: ${e.message}", e)
                            if (!isFinishing) {
                                Toast.makeText(this@DipSelectionActivity, "Error al procesar respuesta: ${e.message}", Toast.LENGTH_SHORT).show()
                                setResult(RESULT_CANCELED)
                                finish()
                            }
                        }
                    }
                }
            })
        }
    }
}