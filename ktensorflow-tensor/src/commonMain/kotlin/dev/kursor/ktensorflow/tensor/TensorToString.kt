package dev.kursor.ktensorflow.tensor

import dev.kursor.ktensorflow.InternalKTensorFlowApi
import dev.kursor.ktensorflow.tensor.impl.contentToString

/** Больше стольких элементов тензор печатается сокращённо - тот же порог, что у NumPy. */
private const val MAX_PRINTED_ELEMENTS = 1000

/** Сколько элементов с каждого края печатается у сокращённого тензора. */
private const val EDGE_ELEMENTS = 3

/**
 * Строковое представление тензора для toString любой его реализации: небольшой тензор печатается
 * целиком, как вложенные массивы, большой - формой, типом и несколькими элементами с краёв.
 *
 * Раньше физический тензор печатал все элементы: кадр 640x640x3 давал строку на 7 млн символов и
 * 0,75 с на iOS, так что случайный лог в цикле кадров останавливал обработку. View и ImageTensor,
 * наоборот, печатали только имя класса.
 */
@InternalKTensorFlowApi
fun Tensor<*>.tensorToString(): String {
    val size = shape.flatSize
    if (size <= MAX_PRINTED_ELEMENTS) {
        return toPhysical().data.contentToString(dataType, shape)
    }

    val head = (0 until EDGE_ELEMENTS).joinToString { getFlat(it).toString() }
    val tail = (size - EDGE_ELEMENTS until size).joinToString { getFlat(it).toString() }
    return "Tensor(shape=$shape, dataType=$dataType, data=[$head, ..., $tail])"
}
