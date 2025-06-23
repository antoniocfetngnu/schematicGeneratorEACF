package com.example.schemaitics

import android.os.Bundle
import android.util.Log
import android.widget.ToggleButton
import androidx.appcompat.app.AppCompatActivity
import com.example.schemaitics.model.Component
import com.example.schemaitics.model.SchematicData
import com.example.schemaitics.utils.DataLoader
import com.google.gson.Gson
import com.google.gson.JsonSyntaxException
import com.google.gson.reflect.TypeToken

class SchematicViewerActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_schematic_viewer)

        val schematicDataJson = intent.getStringExtra("schematic_data") // Cambiar a "SCHEMATIC_JSON" si es necesario
        if (schematicDataJson.isNullOrEmpty()) {
            Log.e("SchematicViewerActivity", "No se recibió schematic_data")
            finish() // Cerrar la actividad si no hay datos
            return
        }

        try {
            val gson = Gson()
            val listType = object : TypeToken<List<Component>>() {}.type
            val components: List<Component> = gson.fromJson(schematicDataJson, listType)
            val rawSchematicData = SchematicData(components)
            val schematicData = DataLoader.processSchematicData(rawSchematicData)

            val schematicView = findViewById<SchematicView>(R.id.schematic_view)
            schematicView.schematicData = schematicData

            val toggleCanvasMove = findViewById<ToggleButton>(R.id.toggle_canvas_move)
            toggleCanvasMove.setOnCheckedChangeListener { _, isChecked ->
                schematicView.isCanvasMoveMode = isChecked
            }
        } catch (e: JsonSyntaxException) {
            Log.e("SchematicViewerActivity", "Error al parsear JSON: ${e.message}")
            finish()
        } catch (e: Exception) {
            Log.e("SchematicViewerActivity", "Error inesperado: ${e.message}")
            finish()
        }
    }
}