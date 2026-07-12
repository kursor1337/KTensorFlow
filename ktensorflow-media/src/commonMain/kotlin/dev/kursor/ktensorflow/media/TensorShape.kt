package dev.kursor.ktensorflow.media

import dev.kursor.ktensorflow.tensor.TensorShape

fun TensorShape(
    n: Int,
    h: Int,
    w: Int,
    c: Int,
    layout: ImageTensorLayout
): TensorShape = with(layout) {
    val dims = IntArray(4)
    dims[nIndex] = n
    dims[cIndex] = c
    dims[hIndex] = h
    dims[wIndex] = w
    TensorShape(*dims)
}