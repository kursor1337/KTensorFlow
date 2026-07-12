package dev.kursor.ktensorflow.vision

fun Image.resizeWithPad(
    targetWidth: Int,
    targetHeight: Int,
    padColorArgb: Int = 0xFF000000.toInt()
): PaddedImage {
    val scale = minOf(
        targetWidth.toFloat() / width.toFloat(),
        targetHeight.toFloat() / height.toFloat()
    )

    val scaledWidth = (width * scale).toInt()
    val scaledHeight = (height * scale).toInt()

    val padX = (targetWidth - scaledWidth) / 2
    val padY = (targetHeight - scaledHeight) / 2

    val scaledImage = this.resize(scaledWidth, scaledHeight)
    val scaledPixels = scaledImage.getPixels()

    val paddedPixels = IntArray(targetWidth * targetHeight) { padColorArgb }

    for (y in 0 until scaledHeight) {
        val srcOffset = y * scaledWidth
        val dstOffset = (y + padY) * targetWidth + padX
        
        scaledPixels.copyInto(
            destination = paddedPixels,
            destinationOffset = dstOffset,
            startIndex = srcOffset,
            endIndex = srcOffset + scaledWidth
        )
    }

    scaledImage.close()

    val finalImage = Image(targetWidth, targetHeight, pixelFormat, paddedPixels)

    return PaddedImage(
        delegate = finalImage,
        info = PadInfo(
            width,
            height,
            targetWidth,
            targetHeight,
            padX,
            padY,
            scale
        )
    )
}