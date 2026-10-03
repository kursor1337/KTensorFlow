package dev.kursor.ktensorflow.vision

/**
 * Represents the layout of dimensions in an image tensor, mapping semantic axes
 * (Batch, Channels, Height, Width) to their respective indices in a tensor's shape.
 *
 * @property nIndex The dimension index representing the batch size (N).
 * @property cIndex The dimension index representing the number of channels (C).
 * @property hIndex The dimension index representing the height (H).
 * @property wIndex The dimension index representing the width (W).
 */
data class ImageTensorLayout(
    val nIndex: Int,
    val cIndex: Int,
    val hIndex: Int,
    val wIndex: Int
) {
    companion object {
        /**
         * Layout where dimensions are ordered as: Batch (N), Channels (C), Height (H), Width (W).
         *
         * Indices: N=0, C=1, H=2, W=3.
         */
        val NCHW = ImageTensorLayout(0, 1, 2, 3)
        /**
         * Layout where dimensions are ordered as: Batch (N), Height (H), Width (W), Channels (C).
         *
         * Indices: N=0, C=3, H=1, W=2.
         */
        val NHWC = ImageTensorLayout(0, 3, 1, 2)

        /**
         * Layout where dimensions are ordered as: Channels (C), Height (H), Width (W), Batch (N).
         *
         * Indices: N=3, C=0, H=1, W=2.
         */
        val CHWN = ImageTensorLayout(3, 0, 1, 2)
    }
}