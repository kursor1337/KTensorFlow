package dev.kursor.ktensorflow.vision

import platform.CoreGraphics.*
import platform.CoreGraphics.kCGBitmapByteOrder32Big
import platform.CoreGraphics.kCGBitmapByteOrder32Little

val PixelFormat.cgBitmapInfo: UInt
    get() = when (this) {
        is PixelFormat.RGBA -> {
            val littleEndian = rIndex != 0

            val alpha = when (aIndex) {
                0 -> if (littleEndian) {
                    CGImageAlphaInfo.kCGImageAlphaPremultipliedLast
                } else {
                    CGImageAlphaInfo.kCGImageAlphaPremultipliedFirst
                }
                3 -> if (littleEndian) {
                    CGImageAlphaInfo.kCGImageAlphaPremultipliedFirst
                } else {
                    CGImageAlphaInfo.kCGImageAlphaPremultipliedLast
                }
                else -> error("Unsupported alpha index: $aIndex")
            }

            val byteOrder = if (littleEndian) {
                kCGBitmapByteOrder32Big
            } else {
                kCGBitmapByteOrder32Little
            }

            alpha.value or byteOrder
        }

        PixelFormat.Grayscale ->
            CGImageAlphaInfo.kCGImageAlphaNone.value

        else -> error("Unsupported PixelFormat for CoreGraphics: $this")
    }
