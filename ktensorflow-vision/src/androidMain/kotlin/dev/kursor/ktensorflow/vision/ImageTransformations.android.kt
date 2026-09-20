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
    .asImage(source = this, pixelFormat = pixelFormat, closeOriginal = closeOriginal)

actual fun Image.crop(
    rect: Rect,
    closeOriginal: Boolean
): Image {
    requireInsideImage(rect)

    return Bitmap
        .createBitmap(
            platformImage,
            rect.left,
            rect.top,
            rect.right - rect.left,
            rect.bottom - rect.top
        )
        .asImage(source = this, pixelFormat = pixelFormat, closeOriginal = closeOriginal)
}

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
    .asImage(source = this, pixelFormat = pixelFormat, closeOriginal = closeOriginal)

actual fun Image.grayscale(
    closeOriginal: Boolean
): Image {
    val grayBitmap = createBitmap(width, height)

    val canvas = Canvas(grayBitmap)
    val paint = Paint()
    val colorMatrix = ColorMatrix().apply { setSaturation(0f) }
    paint.colorFilter = ColorMatrixColorFilter(colorMatrix)

    canvas.drawBitmap(platformImage, 0f, 0f, paint)

    return grayBitmap.asImage(
        source = this,
        pixelFormat = PixelFormat.Grayscale,
        closeOriginal = closeOriginal
    )
}

/**
 * Оборачивает результат преобразования в [AndroidImage], закрывая исходное изображение.
 *
 * Bitmap.scale и Bitmap.createBitmap возвращают ИСХОДНЫЙ bitmap, когда преобразование ничего
 * не меняет: тот же размер при масштабировании, единичная матрица при повороте, вырезание
 * всего изображения целиком. Закрыть оригинал в этом случае означало бы переработать тот
 * самый bitmap, который только что отдали наружу, - и следующее же чтение пикселей упало бы
 * на переработанном bitmap. Поэтому оригинал закрывается, только если результат
 * действительно другой объект.
 */
private fun Bitmap.asImage(
    source: Image,
    pixelFormat: PixelFormat,
    closeOriginal: Boolean
): Image {
    if (closeOriginal && this !== source.platformImage) {
        source.close()
    }
    return AndroidImage(this, pixelFormat)
}
