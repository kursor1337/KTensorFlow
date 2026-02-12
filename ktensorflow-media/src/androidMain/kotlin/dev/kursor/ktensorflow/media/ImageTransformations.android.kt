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
): Image = bitmap
    .scale(newWidth, newHeight)
    .let { Image(it, pixelFormat) }


actual fun Image.crop(rect: Rect): Image = Bitmap
    .createBitmap(
        bitmap,
        rect.left,
        rect.top,
        rect.right - rect.left,
        rect.bottom - rect.top
    )
    .let { Image(it, pixelFormat) }

actual fun Image.rotate(degrees: Float): Image = Bitmap
    .createBitmap(
        bitmap,
        0,
        0,
        bitmap.width,
        bitmap.height,
        Matrix().apply {
            postRotate(degrees)
        },
        true
    )
    .let { Image(it, pixelFormat) }

actual fun Image.grayscale(): Image {
    val grayBitmap = createBitmap(width, height)

    val canvas = Canvas(grayBitmap)
    val paint = Paint()
    val colorMatrix = ColorMatrix().apply { setSaturation(0f) }
    paint.colorFilter = ColorMatrixColorFilter(colorMatrix)

    canvas.drawBitmap(bitmap, 0f, 0f, paint)
    return Image(grayBitmap, PixelFormat.Grayscale)
}
