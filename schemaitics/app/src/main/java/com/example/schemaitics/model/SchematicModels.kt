package com.example.schemaitics.model

data class Pin(
    val pin_num: Int,
    val pin_name: String,
    val connected: Boolean,
    val net: Int? = null,
    val x: Float? = null
)

data class Component(
    val type: String,
    val instance: String,
    val pins: List<Pin>,
    var rotation: Float = 0f // Ángulo en grados, por defecto 0
)

data class SchematicData(
    val components: List<Component>
)