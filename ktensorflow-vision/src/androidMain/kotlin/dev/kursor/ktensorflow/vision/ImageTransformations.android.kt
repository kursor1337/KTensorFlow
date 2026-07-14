package dev.kursor.ktensorflow.vision

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.Matrix
import android.graphics.Paint
import androidx.core.graphics.createBitmap
import androidx.core.graphics.scale

actual fun Image.resize(
    newWidth: Int,
    newHeight: Int,
    closeOriginal: Boolean
): Image = platformImage
    .scale(newWidth, newHeight)
    .let { AndroidImage(it, pixelFormat) }
    .also { if (closeOriginal) close() }


actual fun Image.crop(
    rect: Rect,
    closeOriginal: Boolean
): Image = Bitmap
    .createBitmap(
        platformImage,
        rect.left,
        rect.top,
        rect.right - rect.left,
        rect.bottom - rect.top
    )
    .let { AndroidImage(it, pixelFormat) }
    .also { if (closeOriginal) close() }

actual fun Image.rotate(
    degrees: Float,
    closeOriginal: Boolean
): Image = Bitmap
    .createBitmap(
        platformImage,
        0,
        0,
        platformImage.width,
        platformImage.height,
        Matrix().apply {
            postRotate(degrees)
        },
        true
    )
    .let { AndroidImage(it, pixelFormat) }
    .also { if (closeOriginal) close() }

actual fun Image.grayscale(
    closeOriginal: Boolean
): Image {
    val grayBitmap = createBitmap(width, height)

    val canvas = Canvas(grayBitmap)
    val paint = Paint()
    val colorMatrix = ColorMatrix().apply { setSaturation(0f) }
    paint.colorFilter = ColorMatrixColorFilter(colorMatrix)

    canvas.drawBitmap(platformImage, 0f, 0f, paint)
    if (closeOriginal) close()
    return AndroidImage(grayBitmap, PixelFormat.Grayscale)
}
