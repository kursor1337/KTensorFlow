package dev.kursor.ktensorflow.vision

import kotlinx.cinterop.CPointer
import kotlinx.cinterop.MemScope
import kotlinx.cinterop.Pinned
import kotlinx.cinterop.addressOf
import kotlinx.cinterop.alloc
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.ptr
import kotlinx.cinterop.toCValues
import kotlinx.cinterop.usePinned
import platform.Accelerate.kvImageNoError
import platform.Accelerate.kvImageNoFlags
import platform.Accelerate.vImageConvert_RGB888toRGBA8888
import platform.Accelerate.vImageConvert_RGBA8888toRGB888
import platform.Accelerate.vImageMatrixMultiply_ARGB8888ToPlanar8
import platform.Accelerate.vImageUnpremultiplyData_ARGB8888
import platform.Accelerate.vImageUnpremultiplyData_RGBA8888
import platform.Accelerate.vImage_Buffer
import platform.Accelerate.vImage_Error

// Обёртки над vImage (Accelerate): SIMD-реализации попиксельных операций вместо циклов на Kotlin

/** Разворачивает 3-канальные пиксели в 4 канала с непрозрачной альфой в конце, порядок цвета сохраняется. */
internal fun expandToFourChannels(rgb: ByteArray, width: Int, height: Int): ByteArray {
    val out = ByteArray(width * height * 4)
    vImage(rgb, 3, out, 4, width, height) { src, dst ->
        vImageConvert_RGB888toRGBA8888(src, null, 0xFFu, dst, false, kvImageNoFlags)
    }
    return out
}

/** Отбрасывает четвёртый канал - обратное к [expandToFourChannels]. */
internal fun packToThreeChannels(rgba: ByteArray, out: ByteArray, width: Int, height: Int) {
    vImage(rgba, 4, out, 3, width, height) { src, dst ->
        vImageConvert_RGBA8888toRGB888(src, dst, kvImageNoFlags)
    }
}

/**
 * Яркость по ITU-R 601 из 4-канального буфера в один канал. Веса в 1/32768 долях
 * (9798 + 19235 + 3735 = 32768), половина делителя в post bias округляет к ближайшему.
 */
internal fun lumaRec601(pixels: ByteArray, format: PixelFormat.RGBA, out: ByteArray, width: Int, height: Int) {
    val matrix = ShortArray(4)
    matrix[format.rIndex] = 9798
    matrix[format.gIndex] = 19235
    matrix[format.bIndex] = 3735
    vImage(pixels, 4, out, 1, width, height) { src, dst ->
        vImageMatrixMultiply_ARGB8888ToPlanar8(
            src,
            dst,
            matrix.toCValues(),
            LUMA_DIVISOR,
            null,
            LUMA_DIVISOR / 2,
            kvImageNoFlags
        )
    }
}

private const val LUMA_DIVISOR = 32768

/** Снимает premultiply: CoreGraphics хранит цвет умноженным на альфу. */
internal fun unpremultiply(pixels: ByteArray, format: PixelFormat.RGBA, width: Int, height: Int): ByteArray {
    val out = ByteArray(pixels.size)
    vImage(pixels, 4, out, 4, width, height) { src, dst ->
        // Порядок цветовых каналов vImage не важен, важно только, где альфа
        if (format.aIndex == 0) {
            vImageUnpremultiplyData_ARGB8888(src, dst, kvImageNoFlags)
        } else {
            vImageUnpremultiplyData_RGBA8888(src, dst, kvImageNoFlags)
        }
    }
    return out
}

private inline fun vImage(
    src: ByteArray,
    srcChannels: Int,
    dst: ByteArray,
    dstChannels: Int,
    width: Int,
    height: Int,
    operation: (CPointer<vImage_Buffer>, CPointer<vImage_Buffer>) -> vImage_Error
) {
    // У пустого массива нет адреса для vImage_Buffer, а обрабатывать в нём нечего
    if (width == 0 || height == 0) return
    src.usePinned { pinnedSrc ->
        dst.usePinned { pinnedDst ->
            memScoped {
                val error = operation(
                    buffer(pinnedSrc, width, height, srcChannels),
                    buffer(pinnedDst, width, height, dstChannels)
                )
                check(error == kvImageNoError.toLong()) { "vImage operation failed with error $error" }
            }
        }
    }
}

private fun MemScope.buffer(
    pinned: Pinned<ByteArray>,
    width: Int,
    height: Int,
    channels: Int
): CPointer<vImage_Buffer> = alloc<vImage_Buffer> {
    data = pinned.addressOf(0)
    this.width = width.toULong()
    this.height = height.toULong()
    rowBytes = (width * channels).toULong()
}.ptr
