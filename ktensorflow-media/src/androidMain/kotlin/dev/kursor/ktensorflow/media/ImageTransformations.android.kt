package dev.kursor.ktensorflow.media

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.Matrix
import android.graphics.Paint
import androidx.core.graphics.createBitmap
import androidx.core.graphics.scale
import dev.kursor.ktensorflow.media.camera.Rect

actual fun Image.resize(
    newWidth: Int,
    newHeight: Int
): Image = platformImage
    .scale(newWidth, newHeight)
    .let { AndroidImage(it, pixelFormat) }


actual fun Image.crop(rect: Rect): Image = Bitmap
    .createBitmap(
        platformImage,
        rect.left,
        rect.top,
        rect.right - rect.left,
        rect.bottom - rect.top
    )
    .let { AndroidImage(it, pixelFormat) }

actual fun Image.rotate(degrees: Float): Image = Bitmap
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

actual fun Image.grayscale(): Image {
    val grayBitmap = createBitmap(width, height)

    val canvas = Canvas(grayBitmap)
    val paint = Paint()
    val colorMatrix = ColorMatrix().apply { setSaturation(0f) }
    paint.colorFilter = ColorMatrixColorFilter(colorMatrix)

    canvas.drawBitmap(platformImage, 0f, 0f, paint)
    return AndroidImage(grayBitmap, PixelFormat.Grayscale)
}
