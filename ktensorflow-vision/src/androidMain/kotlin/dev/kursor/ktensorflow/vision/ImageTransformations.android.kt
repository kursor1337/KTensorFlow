package dev.kursor.ktensorflow.vision

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.PorterDuff
import android.graphics.PorterDuffXfermode
import androidx.core.graphics.createBitmap
import androidx.core.graphics.scale
import android.graphics.Rect as AndroidRect

actual fun Image.resize(
    newWidth: Int,
    newHeight: Int,
    closeOriginal: Boolean
): Image {
    requirePositiveSize(newWidth, newHeight)

    return platformImage
        .scale(newWidth, newHeight)
        .asImage(source = this, pixelFormat = pixelFormat, closeOriginal = closeOriginal)
}

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
    paint.colorFilter = ColorMatrixColorFilter(LumaRec601)

    canvas.drawBitmap(platformImage, 0f, 0f, paint)

    return grayBitmap.asImage(
        source = this,
        pixelFormat = PixelFormat.Grayscale,
        closeOriginal = closeOriginal
    )
}

internal actual fun Image.drawLetterboxed(
    scaledWidth: Int,
    scaledHeight: Int,
    targetWidth: Int,
    targetHeight: Int,
    padX: Int,
    padY: Int,
    padColorArgb: Int
): Image {
    val out = createBitmap(targetWidth, targetHeight)
    val canvas = Canvas(out)
    // SRC, а не SRC_OVER: и заливка, и картинка заменяют пиксели, как при копировании массива,
    // иначе полупрозрачные пиксели смешались бы с цветом полей
    canvas.drawColor(padColorArgb, PorterDuff.Mode.SRC)
    val paint = Paint(Paint.FILTER_BITMAP_FLAG).apply {
        xfermode = PorterDuffXfermode(PorterDuff.Mode.SRC)
    }
    canvas.drawBitmap(
        platformImage,
        null,
        AndroidRect(padX, padY, padX + scaledWidth, padY + scaledHeight),
        paint
    )
    return AndroidImage(out, pixelFormat)
}

/**
 * Оборачивает результат преобразования в [AndroidImage] и при необходимости закрывает исходник.
 *
 * Bitmap.scale и Bitmap.createBitmap возвращают ИСХОДНЫЙ bitmap, когда преобразование ничего
 * не меняет: тот же размер при масштабировании, единичная матрица при повороте, вырезание
 * всего изображения целиком. Тогда:
 * - если исходник закрывается, bitmap просто переходит к результату - переработать его значило
 *   бы сломать только что отданное изображение;
 * - если исходник остаётся, результату нужна своя копия: иначе оба изображения делили бы один
 *   bitmap, и закрытие любого ломало бы второе. На iOS эти операции копируют всегда.
 */
private fun Bitmap.asImage(
    source: Image,
    pixelFormat: PixelFormat,
    closeOriginal: Boolean
): Image {
    val bitmap = when {
        this !== source.platformImage -> {
            if (closeOriginal) source.close()
            this
        }
        closeOriginal -> this
        else -> copy(config ?: Bitmap.Config.ARGB_8888, isMutable)
    }
    return AndroidImage(bitmap, pixelFormat)
}

/**
 * Перевод в яркость по ITU-R 601 (0.299, 0.587, 0.114) с сохранением альфы.
 *
 * ColorMatrix.setSaturation(0) для этого не подходит: он считает яркость по Rec.709
 * (0.213, 0.715, 0.072), и та же картинка давала на Android другие значения, чем на iOS и в
 * ImageTensor.grayscale - чистый красный превращался в 54 вместо 76.
 */
private val LumaRec601 = ColorMatrix(
    floatArrayOf(
        0.299f, 0.587f, 0.114f, 0f, 0f,
        0.299f, 0.587f, 0.114f, 0f, 0f,
        0.299f, 0.587f, 0.114f, 0f, 0f,
        0f, 0f, 0f, 1f, 0f,
    )
)
