package dev.kursor.ktensorflow.vision

import dev.kursor.ktensorflow.tensor.TensorShape

/**
 * Creates a 4D [TensorShape] for an image tensor based on the provided dimensions and [ImageTensorLayout].
 *
 * @param n the number of images in the batch (batch size).
 * @param h the height of the image.
 * @param w the width of the image.
 * @param c the number of channels (e.g., 3 for RGB).
 * @param layout the memory layout (NHWC or NCHW) determining the order of the dimensions.
 * @return a [TensorShape] with dimensions mapped according to the specified layout.
 */
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