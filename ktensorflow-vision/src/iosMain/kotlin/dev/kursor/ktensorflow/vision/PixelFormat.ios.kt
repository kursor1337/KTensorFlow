package dev.kursor.ktensorflow.vision

import platform.CoreGraphics.CGImageAlphaInfo
import platform.CoreGraphics.kCGBitmapByteOrder32Big
import platform.CoreGraphics.kCGBitmapByteOrder32Little

internal val PixelFormat.cgBitmapInfo: UInt
    get() = when (this) {
        is PixelFormat.RGBA -> {
            // swapped == true means blue comes before red in memory (BGRA/ABGR),
            // which reverses the visual meaning of "alpha first/last" once combined
            // with the corresponding 32-bit byte order - see Apple's documented
            // CGBitmapInfo combinations: RGBA=Last|Big, ARGB=First|Big,
            // BGRA=First|Little, ABGR=Last|Little.
            val swapped = rIndex > bIndex
            val alphaFirst = (aIndex == 0) xor swapped

            val alpha = if (alphaFirst) {
                CGImageAlphaInfo.kCGImageAlphaPremultipliedFirst
            } else {
                CGImageAlphaInfo.kCGImageAlphaPremultipliedLast
            }

            val byteOrder = if (swapped) {
                kCGBitmapByteOrder32Little
            } else {
                kCGBitmapByteOrder32Big
            }

            alpha.value or byteOrder
        }

        PixelFormat.Grayscale ->
            CGImageAlphaInfo.kCGImageAlphaNone.value

        else -> error("Unsupported PixelFormat for CoreGraphics: $this")
    }
