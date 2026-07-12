package dev.kursor.ktensorflow.vision

import dev.kursor.ktensorflow.tensor.slice
import dev.kursor.ktensorflow.tensor.toFloatTensor
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

    when (val pf = pixelFormat) {
        PixelFormat.Grayscale -> return this
        is PixelFormat.RGB -> {
            for (i in 0..<width) {
                for (j in 0..<height) {
                    for (b in 0..<batches) {
                        val a = rWeight * this[b, j, i, pf.rIndex] +
                                gWeight * this[b, j, i, pf.gIndex] +
                                bWeight * this[b, j, i, pf.bIndex]
                        result[b, j, i, 0] = a
                    }
                }
            }
        }

        is PixelFormat.RGBA -> {
            for (i in 0..<width) {
                for (j in 0..<height) {
                    for (b in 0..<batches) {
                        result[b, j, i, 0] = (rWeight * this[b, j, i, pf.rIndex] +
                                gWeight * this[b, j, i, pf.gIndex] +
                                bWeight * this[b, j, i, pf.bIndex]) *
                                this[b, j, i, pf.aIndex]
                    }
                }
            }
        }
    }

    return result
}

@JvmName("grayscaleUByte")
fun ImageTensor<UByte>.grayscale(): ImageTensor<UByte> {
    val pf = pixelFormat
    if (pf == PixelFormat.Grayscale) return this

    val result = ImageTensor<UByte>(width, height, PixelFormat.Grayscale, layout)

    val rIdx = when (pf) {
        is PixelFormat.RGB -> pf.rIndex
        is PixelFormat.RGBA -> pf.rIndex
    }
    val gIdx = when (pf) {
        is PixelFormat.RGB -> pf.gIndex
        is PixelFormat.RGBA -> pf.gIndex
    }
    val bIdx = when (pf) {
        is PixelFormat.RGB -> pf.bIndex
        is PixelFormat.RGBA -> pf.bIndex
    }

    var dstIdx = 0
    for (n in 0 until batches) {
        for (h in 0 until height) {
            for (w in 0 until width) {
                val r = this[n, h, w, rIdx].toInt() and 0xFF
                val g = this[n, h, w, gIdx].toInt() and 0xFF
                val b = this[n, h, w, bIdx].toInt() and 0xFF

                val gray = (r * 77 + g * 150 + b * 29) shr 8
                result.setFlat(dstIdx++, gray.toUByte())
            }
        }
    }
    return result
}


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
            for (nb in 0 until batches) {

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
                    val Q11 = get(nb, y1, x1, c) // Top-Left
                    val Q21 = get(nb, y1, x2, c) // Top-Right
                    val Q12 = get(nb, y2, x1, c) // Bottom-Left
                    val Q22 = get(nb, y2, x2, c) // Bottom-Right

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
    }

    return ImageTensor(layout, pixelFormat, resizedData)
}

@JvmName("resizeUByte")
fun ImageTensor<UByte>.resize(newWidth: Int, newHeight: Int): ImageTensor<UByte> {
    if (newWidth == width && newHeight == height) return this

    val result = ImageTensor<UByte>(newWidth, newHeight, pixelFormat, layout)
    val xRatio = width.toFloat() / newWidth
    val yRatio = height.toFloat() / newHeight

    // For UInt8 use Nearest Neighbor,
    // since bilinear interpolation requires converting to float
    for (n in 0 until batches) {
        for (h in 0 until newHeight) {
            val srcH = (h * yRatio).toInt()
            for (w in 0 until newWidth) {
                val srcW = (w * xRatio).toInt()
                for (c in 0 until channels) {
                    result[n, h, w, c] = this[n, srcH, srcW, c]
                }
            }
        }
    }
    return result
}

fun ImageTensor<UByte>.toFloatImageTensor(): ImageTensor<Float> =
    toFloatTensor()
        .toImageTensor(pixelFormat, layout)

fun ImageTensor<Float>.crop(rect: Rect): ImageTensor<Float> = this
    .slice(
        arrayOf(
            rect.left..<rect.right,
            rect.top..<rect.bottom,
            0..<channels
        )
    )
    .toImageTensor(this.pixelFormat, layout)

@JvmName("cropUByte")
fun ImageTensor<UByte>.crop(rect: Rect): ImageTensor<UByte> = this
    .slice(
        arrayOf(
            rect.left..<rect.right,
            rect.top..<rect.bottom,
            0..<channels
        )
    )
    .toImageTensor(this.pixelFormat, layout)