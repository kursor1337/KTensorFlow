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

//fun indexOf(
//    n: Int,
//    c: Int,
//    h: Int,
//    w: Int,
//    shape: TensorShape,
//    layout: ImageTensorLayout,
//): Int = with(layout) {
//    val coords = IntArray(4)
//    coords[nIndex] = n
//    coords[cIndex] = c
//    coords[hIndex] = h
//    coords[wIndex] = w
//    return shape.linearIndex(coords)
//}