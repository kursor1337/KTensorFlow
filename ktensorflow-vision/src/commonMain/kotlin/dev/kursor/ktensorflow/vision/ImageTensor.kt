package dev.kursor.ktensorflow.vision

import dev.kursor.ktensorflow.tensor.Tensor
import dev.kursor.ktensorflow.tensor.TensorDataType

/**
 * Represents a [Tensor] specialized for image data, providing structured access to dimensions
 */
interface ImageTensor<T : Any> : Tensor<T> {

    /**
     * The pixel format of the image data, defining the color space and channel organization (e.g., RGB, RGBA).
     *
     * @see PixelFormat
     */
    val pixelFormat: PixelFormat

    /**
     * The layout of the image data, defining the order of dimensions (e.g., NHWC, NCHW).
     *
     * @see ImageTensorLayout
     */
    val layout: ImageTensorLayout

    /**
     * The batch size of the image data, defining the number of images in a batch.
     */
    val batch: Int

    /**
     * The width of the image data, defining the horizontal dimension.
     */
    val width: Int

    /**
     * The height of the image data, defining the vertical dimension.
     */
    val height: Int

    /**
     * The number of channels in the image data, defining the color depth.
     * RGB - 3, RGBA - 4, Grayscale - 1
     */
    val channels: Int

    /**
     * Returns the value at the specified [n] batch, [h] height, [w] width, and [c] channel coordinates.
     *
     * @param n the batch index.
     * @param h the height (y-coordinate) index.
     * @param w the width (x-coordinate) index.
     * @param c the channel index.
     * @return the value of type [T] at the given coordinates.
     */
    operator fun get(n: Int, h: Int, w: Int, c: Int): T

    /**
     * Returns the value at the specified [h] height, [w] width, and [c] channel coordinates.
     * Batch coordinate used is 0
     *
     * @param h the height (y-coordinate) index.
     * @param w the width (x-coordinate) index.
     * @param c the channel index.
     * @return the value of type [T] at the given coordinates.
     */
    operator fun get(h: Int, w: Int, c: Int) = get(0, h, w, c)


    /**
     * Sets the value at the specified batch index [n], height [h], width [w], and channel [c].
     *
     * @param n the batch index
     * @param h the height index
     * @param w the width index
     * @param c the channel index
     * @param value the value to set at the specified position
     */
    operator fun set(n: Int, h: Int, w: Int, c: Int, value: T)

    /**
     * Sets the value at the specified height index [h], width [w], and channel [c].
     * Batch coorginate used is 0
     *
     * @param h the height index
     * @param w the width index
     * @param c the channel index
     * @param value the value to set at the specified position
     */
    operator fun set(h: Int, w: Int, c: Int, value: T) = set(0, h, w, c, value)
}

/**
 * Creates an [ImageTensor] from a [Tensor] with specified [pixelFormat] and [layout].
 *
 * @param tensor the [Tensor] to create the [ImageTensor] from
 * @param pixelFormat the [PixelFormat] of the image data
 * @param layout the [ImageTensorLayout] of the image data
 * @return the created [ImageTensor]
 */
fun <T : Any> ImageTensor(
    tensor: Tensor<T>,
    pixelFormat: PixelFormat,
    layout: ImageTensorLayout
): ImageTensor<T> = ImageTensorImpl(
    tensor = tensor,
    pixelFormat = pixelFormat,
    layout = layout
)

/**
 * Creates a new [ImageTensor] with the specified dimensions, pixel format, and data type.
 *
 * @param T The data type of the tensor elements.
 * @param width The width of the image(s) in pixels.
 * @param height The height of the image(s) in pixels.
 * @param pixelFormat The [PixelFormat] representing the arrangement and number of color channels.
 * @param layout The [ImageTensorLayout] specifying the order of dimensions (defaults to [ImageTensorLayout.NHWC]).
 * @param batchSize The number of images contained in the tensor (defaults to 1).
 * @return A new [ImageTensor] instance of type [T].
 */
inline fun <reified T : Any> ImageTensor(
    width: Int,
    height: Int,
    pixelFormat: PixelFormat,
    layout: ImageTensorLayout = ImageTensorLayout.NHWC,
    batchSize: Int = 1
): ImageTensor<T> = ImageTensor(
    tensor = Tensor(
        dataType = TensorDataType.of<T>(),
        shape = TensorShape(
            n = batchSize,
            h = height,
            w = width,
            c = pixelFormat.channels,
            layout = layout
        )
    ),
    pixelFormat = pixelFormat,
    layout = layout
)

/**
 * Creates a new [ImageTensor] with the specified dimensions, pixel format, and data type.
 *
 * @param T The data type of the tensor elements.
 * @param width The width of the image(s) in pixels.
 * @param height The height of the image(s) in pixels.
 * @param dataType The [TensorDataType] of the tensor elements
 * @param pixelFormat The [PixelFormat] representing the arrangement and number of color channels.
 * @param layout The [ImageTensorLayout] specifying the order of dimensions (defaults to [ImageTensorLayout.NHWC]).
 * @param batchSize The number of images contained in the tensor (defaults to 1).
 * @return A new [ImageTensor] instance of type [T].
 */
fun <T : Any> ImageTensor(
    width: Int,
    height: Int,
    dataType: TensorDataType<T>,
    pixelFormat: PixelFormat,
    layout: ImageTensorLayout = ImageTensorLayout.NHWC,
    batchSize: Int = 1
) = ImageTensor(
    tensor = Tensor(
        dataType = dataType,
        shape = TensorShape(
            n = batchSize,
            h = height,
            w = width,
            c = pixelFormat.channels,
            layout = layout
        )
    ),
    pixelFormat = pixelFormat,
    layout = layout
)

/**
 * Creates an [ImageTensor] of [Float] elements from a 3D array of data.
 *
 * @param layout The memory layout of the image dimensions (e.g., NHWC or NCHW).
 * @param pixelFormat The color format of the pixels (e.g., RGB, RGBA).
 * @param data The raw image data as a 3D array, typically representing height, width, and channels.
 * @return An [ImageTensor] containing the provided float data.
 */
fun ImageTensor(
    layout: ImageTensorLayout,
    pixelFormat: PixelFormat,
    data: Array<Array<FloatArray>>
) = ImageTensor(
    tensor = Tensor<Float>(data),
    pixelFormat = pixelFormat,
    layout = layout
)

/**
 * Converts a [Tensor] to an [ImageTensor].
 * If this [Tensor] is already an [ImageTensor] with the same
 * [PixelFormat] and [ImageTensorLayout], it is returned as is.
 * Otherwise, a new [ImageTensor] is created from the [Tensor].
 *
 * @param pixelFormat The [PixelFormat] of the image data.
 * @param layout The [ImageTensorLayout] of the image data.
 * @return The [ImageTensor] created from the [Tensor].
 */
fun <T : Any> Tensor<T>.toImageTensor(
    pixelFormat: PixelFormat,
    layout: ImageTensorLayout
): ImageTensor<T> {
    return (this as? ImageTensor<T>)
        ?.let {
            if (it.pixelFormat == pixelFormat && it.layout == layout) {
                it
            } else {
                null
            }
        }
        ?: ImageTensor(this, pixelFormat, layout)
}
