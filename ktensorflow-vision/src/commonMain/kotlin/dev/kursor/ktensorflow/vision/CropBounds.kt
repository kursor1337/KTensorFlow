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

/**
 * Проверяет, что в [pixels] ровно по одному значению на пиксель изображения [width] x [height].
 *
 * Без проверки платформы расходились зеркально, и на каждой одна из сторон была тихой: iOS молча
 * создавал изображение из короткого массива, дополняя его прозрачно-чёрными пикселями, а Android
 * молча отбрасывал лишний хвост длинного.
 */
internal fun requirePixelCount(width: Int, height: Int, pixels: IntArray) {
    require(pixels.size == width * height) {
        "A ${width}x$height image needs ${width * height} pixels, got ${pixels.size}"
    }
}

/**
 * Проверяет буфер для getPixels: по контракту он не меньше width * height, лишний хвост не
 * трогается. iOS раньше обходил весь буфер: больший падал, а меньший молча заполнялся частично.
 */
internal fun Image.requirePixelBuffer(buffer: IntArray) {
    require(buffer.size >= width * height) {
        "A buffer for a ${width}x$height image needs at least ${width * height} pixels, got ${buffer.size}"
    }
}
