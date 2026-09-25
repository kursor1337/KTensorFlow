package dev.kursor.ktensorflow.vision

/**
 * Проверяет, что [rect] действительно лежит внутри изображения.
 *
 * Без общей проверки платформы падали по-разному: Bitmap.createBitmap бросал
 * IllegalArgumentException, а CGImageCreateWithImageInRect возвращал null, и вызов
 * заканчивался IllegalStateException. Ловить такое в общем коде было нечем.
 */
internal fun Image.requireInsideImage(rect: Rect) {
    require(rect.right > rect.left && rect.bottom > rect.top) {
        "Crop rect $rect is empty"
    }
    require(
        rect.left >= 0 && rect.top >= 0 && rect.right <= width && rect.bottom <= height
    ) {
        "Crop rect $rect is outside the ${width}x$height image"
    }
}

/**
 * Проверяет размер результата resize. Как и с кропом, без общей проверки платформы падали
 * по-разному: Bitmap бросал IllegalArgumentException, а на iOS нулевой размер доходил до
 * CGBitmapContext и заканчивался выходом за пустой массив.
 */
internal fun requirePositiveSize(width: Int, height: Int) {
    require(width > 0 && height > 0) { "New size must be positive, got ${width}x$height" }
}
