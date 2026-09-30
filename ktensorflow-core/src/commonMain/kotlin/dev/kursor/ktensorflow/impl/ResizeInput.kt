package dev.kursor.ktensorflow.impl

import dev.kursor.ktensorflow.TensorFlowException

/**
 * Меняет форму входа [index] на [dims] так, чтобы неудача не оставляла интерпретатор сломанным.
 *
 * Раньше форма уходила в TensorFlow Lite как есть, и платформы расходились: iOS молча принимал
 * форму, с которой выходы модели получали нулевую размерность, и падал уже на следующем run или
 * getModelMeta; Android бросал исключение, но оставлял вход в новой форме, а после отрицательной
 * размерности интерпретатор не восстанавливался даже возвратом к исходной форме.
 *
 * Теперь форма проверяется заранее (непустая, все размерности положительны - то же требует
 * TensorFlowLiteObjC), а при любом отказе вход возвращается к прежней форме и исключение
 * пробрасывается. Отказом считается и форма, после которой выходы, бывшие корректными,
 * получили недопустимую форму.
 *
 * @param currentShape текущая форма входа или null, если её нельзя прочитать - тогда откатывать
 * некуда.
 * @param outputShapesValid все ли выходы имеют форму с положительными размерностями.
 * @param resizeAndAllocate меняет форму входа и заново выделяет тензоры.
 */
internal inline fun resizeInputSafely(
    index: Int,
    dims: IntArray,
    currentShape: () -> IntArray?,
    outputShapesValid: () -> Boolean,
    resizeAndAllocate: (IntArray) -> Unit
) {
    if (dims.isEmpty() || dims.any { it <= 0 }) {
        throw TensorFlowException(
            "Invalid shape ${dims.contentToString()} for input $index: " +
                "it must have at least one dimension and every dimension must be positive"
        )
    }

    val previousShape = currentShape()
    val outputsWereValid = outputShapesValid()
    try {
        resizeAndAllocate(dims)
        // Только если выходы были корректны до этого: у модели с изначально недопустимой формой
        // выхода (скаляр, динамический тензор) иначе нельзя было бы поменять вход вовсе
        if (outputsWereValid && !outputShapesValid()) {
            throw TensorFlowException(
                "Input $index does not fit shape ${dims.contentToString()}: " +
                    "the model outputs get an invalid shape"
            )
        }
    } catch (e: TensorFlowException) {
        if (previousShape != null) {
            try {
                resizeAndAllocate(previousShape)
            } catch (restoreError: TensorFlowException) {
                e.addSuppressed(restoreError)
            }
        }
        throw e
    }
}

/** Форма, с которой тензор пригоден к работе: хотя бы одна размерность, и все положительны. */
internal fun IntArray.isValidShape(): Boolean = isNotEmpty() && all { it > 0 }
