package dev.kursor.ktensorflow.vision

data class ImageTensorLayout(
    val nIndex: Int,
    val cIndex: Int,
    val hIndex: Int,
    val wIndex: Int
) {
    companion object {
        val NCHW = ImageTensorLayout(0, 1, 2, 3)
        val NHWC = ImageTensorLayout(0, 3, 1, 2)
        val CHWN = ImageTensorLayout(3, 0, 1, 2)
    }
}