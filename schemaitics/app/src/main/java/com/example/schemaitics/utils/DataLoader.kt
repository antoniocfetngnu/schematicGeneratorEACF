package com.example.schemaitics.utils

import com.example.schemaitics.model.Component
import com.example.schemaitics.model.Pin
import com.example.schemaitics.model.SchematicData

object DataLoader {
    fun processSchematicData(rawData: SchematicData): SchematicData {
        val newComponents = mutableListOf<Component>()

        for (comp in rawData.components) {
            when (comp.type) {
                "7404" -> {
                    val grouped = mutableMapOf<Int, Pair<Pin?, Pin?>>()
                    var vccPin: Pin? = null
                    var gndPin: Pin? = null

                    for (pin in comp.pins) {
                        val name = pin.pin_name.uppercase()
                        when {
                            name.startsWith("A") && name.length > 1 -> {
                                val index = name.substring(1).toIntOrNull() ?: continue
                                grouped[index] = grouped.getOrDefault(index, Pair(null, null)).copy(first = pin)
                            }
                            name.startsWith("Y") && name.length > 1 -> {
                                val index = name.substring(1).toIntOrNull() ?: continue
                                grouped[index] = grouped.getOrDefault(index, Pair(null, null)).copy(second = pin)
                            }
                            name == "VCC" -> vccPin = pin
                            name == "GND" -> gndPin = pin
                        }
                    }

                    for ((index, pair) in grouped) {
                        val (aPin, yPin) = pair
                        if ((aPin?.connected == true) || (yPin?.connected == true)) {
                            val pins = listOfNotNull(aPin, yPin)
                            newComponents.add(
                                Component(
                                    instance = "${comp.instance}:$index",
                                    type = "7404_gate",
                                    pins = pins,
                                    rotation = 0f
                                )
                            )
                        }
                    }

                    if ((vccPin?.connected == true) || (gndPin?.connected == true)) {
                        val pins = listOfNotNull(vccPin, gndPin)
                        newComponents.add(
                            Component(
                                instance = "${comp.instance}:power",
                                type = "7404_power",
                                pins = pins,
                                rotation = 0f
                            )
                        )
                    }
                }
                "7408" -> {
                    val grouped = mutableMapOf<Int, Triple<Pin?, Pin?, Pin?>>()
                    var vccPin: Pin? = null
                    var gndPin: Pin? = null

                    for (pin in comp.pins) {
                        val name = pin.pin_name.uppercase()
                        when {
                            name.startsWith("A") && name.length > 1 -> {
                                val index = name.substring(1).toIntOrNull() ?: continue
                                grouped[index] = grouped.getOrDefault(index, Triple(null, null, null)).copy(first = pin)
                            }
                            name.startsWith("B") && name.length > 1 -> {
                                val index = name.substring(1).toIntOrNull() ?: continue
                                grouped[index] = grouped.getOrDefault(index, Triple(null, null, null)).copy(second = pin)
                            }
                            name.startsWith("Y") && name.length > 1 -> {
                                val index = name.substring(1).toIntOrNull() ?: continue
                                grouped[index] = grouped.getOrDefault(index, Triple(null, null, null)).copy(third = pin)
                            }
                            name == "VCC" -> vccPin = pin
                            name == "GND" -> gndPin = pin
                        }
                    }

                    for ((index, triple) in grouped) {
                        val (aPin, bPin, yPin) = triple
                        if ((aPin?.connected == true) || (bPin?.connected == true) || (yPin?.connected == true)) {
                            val pins = listOfNotNull(aPin, bPin, yPin)
                            newComponents.add(
                                Component(
                                    instance = "${comp.instance}:$index",
                                    type = "7408_gate",
                                    pins = pins,
                                    rotation = 0f
                                )
                            )
                        }
                    }

                    if ((vccPin?.connected == true) || (gndPin?.connected == true)) {
                        val pins = listOfNotNull(vccPin, gndPin)
                        newComponents.add(
                            Component(
                                instance = "${comp.instance}:power",
                                type = "7408_power",
                                pins = pins,
                                rotation = 0f
                            )
                        )
                    }
                }
                "7432" -> {
                    val grouped = mutableMapOf<Int, Triple<Pin?, Pin?, Pin?>>()
                    var vccPin: Pin? = null
                    var gndPin: Pin? = null

                    for (pin in comp.pins) {
                        val name = pin.pin_name.uppercase()
                        when {
                            name.startsWith("A") && name.length > 1 -> {
                                val index = name.substring(1).toIntOrNull() ?: continue
                                grouped[index] = grouped.getOrDefault(index, Triple(null, null, null)).copy(first = pin)
                            }
                            name.startsWith("B") && name.length > 1 -> {
                                val index = name.substring(1).toIntOrNull() ?: continue
                                grouped[index] = grouped.getOrDefault(index, Triple(null, null, null)).copy(second = pin)
                            }
                            name.startsWith("Y") && name.length > 1 -> {
                                val index = name.substring(1).toIntOrNull() ?: continue
                                grouped[index] = grouped.getOrDefault(index, Triple(null, null, null)).copy(third = pin)
                            }
                            name == "VCC" -> vccPin = pin
                            name == "GND" -> gndPin = pin
                        }
                    }

                    for ((index, triple) in grouped) {
                        val (aPin, bPin, yPin) = triple
                        if ((aPin?.connected == true) || (bPin?.connected == true) || (yPin?.connected == true)) {
                            val pins = listOfNotNull(aPin, bPin, yPin)
                            newComponents.add(
                                Component(
                                    instance = "${comp.instance}:$index",
                                    type = "7432_gate",
                                    pins = pins,
                                    rotation = 0f
                                )
                            )
                        }
                    }

                    if ((vccPin?.connected == true) || (gndPin?.connected == true)) {
                        val pins = listOfNotNull(vccPin, gndPin)
                        newComponents.add(
                            Component(
                                instance = "${comp.instance}:power",
                                type = "7432_power",
                                pins = pins,
                                rotation = 0f
                            )
                        )
                    }
                }
                else -> newComponents.add(
                    Component(
                        type = comp.type,
                        instance = comp.instance.toString(),
                        pins = comp.pins,
                        rotation = comp.rotation
                    )
                )
            }
        }

        return SchematicData(components = newComponents)
    }
}