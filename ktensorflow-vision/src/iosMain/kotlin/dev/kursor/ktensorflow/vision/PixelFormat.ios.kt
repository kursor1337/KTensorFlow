package dev.kursor.ktensorflow.vision

import platform.CoreGraphics.CGImageAlphaInfo
import platform.CoreGraphics.kCGBitmapByteOrder32Big
import platform.CoreGraphics.kCGBitmapByteOrder32Little

internal val PixelFormat.cgBitmapInfo: UInt
    get() = when (this) {
        is PixelFormat.RGBA -> {
            check(isCoreGraphicsLayout) { "CoreGraphics cannot read $this, it must be converted to $coreGraphicsFormat" }
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

        is PixelFormat.RGB ->
            error("CoreGraphics has no 3-channel layouts, $this must be converted to $coreGraphicsFormat")
    }

/**
 * Раскладки, которые CoreGraphics умеет читать и рисовать: четыре стандартных 4-канальных порядка
 * и серый. Остальные перестановки [cgBitmapInfo] описать не может - по двум признакам (что раньше,
 * R или B, и где альфа) он принял бы, например, R-B-G-A за RGBA, и каналы молча переставились бы.
 */
internal val PixelFormat.isCoreGraphicsLayout: Boolean
    get() = when (this) {
        PixelFormat.Grayscale -> true
        is PixelFormat.RGB -> false
        is PixelFormat.RGBA -> this == PixelFormat.RGBA || this == PixelFormat.ARGB ||
            this == PixelFormat.BGRA || this == PixelFormat.ABGR
    }

/**
 * Формат, в котором пиксели отдаются CoreGraphics: 3 канала дополняются альфой в конце
 * (RGB -> RGBA, BGR -> BGRA), а нестандартный порядок переставляется в RGBA.
 */
internal val PixelFormat.coreGraphicsFormat: PixelFormat
    get() {
        val fourChannels = if (this is PixelFormat.RGB) withAlpha else this
        return if (fourChannels.isCoreGraphicsLayout) fourChannels else PixelFormat.Companion.RGBA
    }

/** Та же раскладка с альфой в конце, как её дописывает [expandToFourChannels]: RGB -> RGBA, BGR -> BGRA. */
internal val PixelFormat.RGB.withAlpha: PixelFormat.RGBA
    get() = PixelFormat.RGBA(rIndex, gIndex, bIndex, 3)

/** Пиксели [bytes] в формате [format], переведённые в раскладку [coreGraphicsFormat]. */
internal fun toCoreGraphicsLayout(bytes: ByteArray, format: PixelFormat, width: Int, height: Int): ByteArray {
    if (format.isCoreGraphicsLayout) return bytes
    val fourChannels = if (format is PixelFormat.RGB) expandToFourChannels(bytes, width, height) else bytes
    val fourFormat = if (format is PixelFormat.RGB) format.withAlpha else format as PixelFormat.RGBA
    if (fourFormat.isCoreGraphicsLayout) return fourChannels
    // Канал i стандартного RGBA берётся из канала исходника с тем же цветом
    val source = intArrayOf(fourFormat.rIndex, fourFormat.gIndex, fourFormat.bIndex, fourFormat.aIndex)
    return permuteChannels(fourChannels, source, width, height)
}

/** Обратное к [toCoreGraphicsLayout]: пиксели в раскладке [coreGraphicsFormat] записываются в [out]. */
internal fun fromCoreGraphicsLayout(
    coreGraphics: ByteArray,
    out: ByteArray,
    format: PixelFormat,
    width: Int,
    height: Int
) {
    val fourFormat = if (format is PixelFormat.RGB) format.withAlpha else format as PixelFormat.RGBA
    val fourChannels = if (fourFormat.isCoreGraphicsLayout) {
        coreGraphics
    } else {
        // Канал fourFormat.rIndex результата берётся из R стандартного RGBA, и так далее
        val source = IntArray(4)
        source[fourFormat.rIndex] = 0
        source[fourFormat.gIndex] = 1
        source[fourFormat.bIndex] = 2
        source[fourFormat.aIndex] = 3
        permuteChannels(coreGraphics, source, width, height)
    }
    if (format is PixelFormat.RGB) {
        packToThreeChannels(fourChannels, out, width, height)
    } else {
        fourChannels.copyInto(out)
    }
}
