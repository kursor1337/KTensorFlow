package dev.kursor.ktensorflow.vision

import dev.kursor.ktensorflow.tensor.strides

/**
 * Смещения элементов тензора изображения, посчитанные один раз на всю тензоризацию.
 *
 * Обход идёт построчно и по столбцам с наращиванием смещения, поэтому позиция пикселя
 * не пересчитывается по страйдам на каждое обращение к каналу, а деление и остаток от
 * деления для восстановления координат из плоского индекса вообще не нужны. Так один
 * и тот же код одинаково быстро работает для любого [ImageTensorLayout] - и для NHWC,
 * и для NCHW, и для пользовательского, - а также для любого элемента батча.
 *
 * На этом обходе построены все преобразования изображения в тензор и обратно, а также
 * grayscale и resize над самим тензором.
 */
internal class ImageOffsets(tensor: ImageTensor<*>) {

    /** Расстояние между соседними изображениями батча. */
    val batch: Int

    /** Расстояние между соседними строками изображения. */
    val row: Int

    /** Расстояние между соседними пикселями строки. */
    val column: Int

    /** Расстояние между соседними каналами одного пикселя. */
    val channel: Int

    val width: Int = tensor.width

    val height: Int = tensor.height

    init {
        val strides = tensor.shape.strides()
        val layout = tensor.layout

        batch = strides[layout.nIndex]
        row = strides[layout.hIndex]
        column = strides[layout.wIndex]
        channel = strides[layout.cIndex]
    }
}
