package dev.kursor.ktensorflow.media

import dev.kursor.ktensorflow.media.camera.Rect
import dev.kursor.ktensorflow.tensor.get
import dev.kursor.ktensorflow.tensor.mapInPlace
import dev.kursor.ktensorflow.tensor.normalize
import dev.kursor.ktensorflow.tensor.set
import dev.kursor.ktensorflow.tensor.slice
import dev.kursor.ktensorflow.tensor.times
import dev.kursor.ktensorflow.tensor.toFloatTensor
import dev.kursor.ktensorflow.tensor.toUByteTensor
import kotlin.jvm.JvmName

fun ImageTensor<Float>.grayscale(
    rWeight: Float = 0.299f,
    gWeight: Float = 0.587f,
    bWeight: Float = 0.114f
): ImageTensor<Float> {

    val result = ImageTensor(
        width = width,
        height = height,
        dataType = dataType,
        pixelFormat = PixelFormat.Grayscale
    )

    when (pixelFormat) {
        PixelFormat.Grayscale -> return this
        is PixelFormat.RGB -> {
            for (i in 0..<width) {
                for (j in 0..<height) {
                    result[i, j, 0] = rWeight * this[i, j, pixelFormat.rIndex] +
                            gWeight * this[i, j, pixelFormat.gIndex] +
                            bWeight * this[i, j, pixelFormat.bIndex]
                }
            }
        }

        is PixelFormat.RGBA -> {
            for (i in 0..<width) {
                for (j in 0..<height) {
                    result[i, j, 0] = (rWeight * this[i, j, pixelFormat.rIndex] +
                            gWeight * this[i, j, pixelFormat.gIndex] +
                            bWeight * this[i, j, pixelFormat.bIndex]) *
                            this[i, j, pixelFormat.aIndex]
                }
            }
        }
    }

    return result
}

@JvmName("grayscaleUByte")
fun ImageTensor<UByte>.grayscale(
    rWeight: Float = 0.299f,
    gWeight: Float = 0.587f,
    bWeight: Float = 0.114f
): ImageTensor<UByte> = this
    .toFloatTensor()
    .normalize()
    .toImageTensor(pixelFormat)
    .grayscale(rWeight, gWeight, bWeight)
    .times(255f)
    .also { tensor -> tensor.mapInPlace { it.coerceIn(0f, 255f) } }
    .toUByteTensor()
    .toImageTensor(pixelFormat)


/**
 * Resizes the given Picture to the new specified dimensions using Bilinear Interpolation.
 *
 * Bilinear interpolation calculates the value of the new pixel based on a weighted average
 * of the four nearest original pixels, providing a smoother result than nearest-neighbor
 * resizing.
 *
 * @param newWidth The desired width of the resulting picture.
 * @param newHeight The desired height of the resulting picture.
 * @return A new Picture object with the resized dimensions and interpolated data.
 */
fun ImageTensor<Float>.resize(newWidth: Int, newHeight: Int): ImageTensor<Float> {
    if (newWidth <= 0 || newHeight <= 0) {
        throw IllegalArgumentException("New dimensions must be positive.")
    }

    if (newWidth == width && newHeight == height) {
        return this // No resize needed
    }

    val resizedData = Array(newWidth) { Array(newHeight) { FloatArray(channels) } }

    // Calculate scale factors
    // We use (old - 1) / (new - 1) ratio for boundary alignment
    val xRatio = (width - 1).toFloat() / (newWidth - 1).toFloat()
    val yRatio = (height - 1).toFloat() / (newHeight - 1).toFloat()

    for (ny in 0 until newHeight) {
        for (nx in 0 until newWidth) {

            // 1. Calculate the corresponding float coordinates in the original image
            val ox = nx * xRatio
            val oy = ny * yRatio

            // 2. Find the four surrounding original pixels (Q11, Q21, Q12, Q22)
            val x1 = ox.toInt()
            val y1 = oy.toInt()

            // x2 and y2 are min(x1 + 1, max_index)
            val x2 = (x1 + 1).coerceAtMost(width - 1)
            val y2 = (y1 + 1).coerceAtMost(height - 1)

            // Fractional parts (interpolation weights)
            val dx = ox - x1
            val dy = oy - y1

            // Loop through all channels (R, G, B, A, etc.)
            for (c in 0 until channels) {
                // Get the four surrounding pixel component values
                val Q11 = get(x1, y1, c) // Top-Left
                val Q21 = get(x2, y1, c) // Top-Right
                val Q12 = get(x1, y2, c) // Bottom-Left
                val Q22 = get(x2, y2, c) // Bottom-Right

                // 3. Horizontal Interpolation (Linear)
                // R1 = interpolate(Q11, Q21) -> Interpolated value on top edge (y1)
                val R1 = Q11 * (1f - dx) + Q21 * dx
                // R2 = interpolate(Q12, Q22) -> Interpolated value on bottom edge (y2)
                val R2 = Q12 * (1f - dx) + Q22 * dx

                // 4. Vertical Interpolation (Bilinear)
                // P' = interpolate(R1, R2) -> Final interpolated value
                val P_prime = R1 * (1f - dy) + R2 * dy

                // Clamp the final value (e.g., in case of floating point errors) and store it
                resizedData[ny][nx][c] = P_prime.coerceIn(0f, 1f)
            }
        }
    }

    return ImageTensor(pixelFormat, resizedData)
}

@JvmName("resizeUByte")
fun ImageTensor<UByte>.resize(newWidth: Int, newHeight: Int): ImageTensor<UByte> = this
    .toFloatTensor()
    .normalize()
    .toImageTensor(pixelFormat)
    .resize(newWidth, newHeight)
    .times(255f)
    .also { tensor -> tensor.mapInPlace { it.coerceIn(0f, 255f) } }
    .toUByteTensor()
    .toImageTensor(pixelFormat)

fun ImageTensor<UByte>.toFloatImageTensor(): ImageTensor<Float> =
    toFloatTensor()
        .toImageTensor(pixelFormat)

fun ImageTensor<Float>.crop(rect: Rect): ImageTensor<Float> = this
    .slice(
        arrayOf(
            rect.left..rect.right,
            rect.top..rect.bottom,
            0..channels
        )
    )
    .toImageTensor(this.pixelFormat)

@JvmName("cropUByte")
fun ImageTensor<UByte>.crop(rect: Rect): ImageTensor<UByte> = this
    .toFloatImageTensor()
    .crop(rect)
    .toUByteTensor()
    .toImageTensor(this.pixelFormat)