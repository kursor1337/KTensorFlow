package dev.kursor.ktensorflow.impl

import dev.kursor.ktensorflow.TensorFlowException
import java.io.IOException

/**
 * Выполняет вызов к TensorFlow Lite, приводя его отказ к [TensorFlowException].
 *
 * Java-API TensorFlow Lite не имеет собственного типа исключения и бросает наружу
 * IllegalArgumentException, IllegalStateException или IOException. На iOS тот же отказ
 * приходит как NSError и уже превращается в [TensorFlowException], поэтому без обёртки
 * поймать ошибку загрузки модели или инференса из общего кода было нечем.
 *
 * Оборачивается только сам вызов в рантайм, а не окружающий его код, чтобы обёртка не
 * проглатывала ошибки самой библиотеки.
 *
 * @param action Что именно не удалось - попадает в сообщение исключения.
 */
internal inline fun <T> tensorFlowCall(action: String, block: () -> T): T =
    try {
        block()
    } catch (e: TensorFlowException) {
        throw e
    } catch (e: IOException) {
        throw TensorFlowException("Failed to $action", e)
    } catch (e: RuntimeException) {
        throw TensorFlowException("Failed to $action", e)
    }
