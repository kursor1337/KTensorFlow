package tensor

import dev.kursor.ktensorflow.tensor.PhysicalTensor
import dev.kursor.ktensorflow.tensor.Tensor
import dev.kursor.ktensorflow.tensor.TensorShape
import dev.kursor.ktensorflow.tensor.TensorView
import dev.kursor.ktensorflow.tensor.flatten
import dev.kursor.ktensorflow.tensor.get
import dev.kursor.ktensorflow.tensor.permuted
import dev.kursor.ktensorflow.tensor.reshape
import dev.kursor.ktensorflow.tensor.set
import dev.kursor.ktensorflow.tensor.slice
import dev.kursor.ktensorflow.tensor.squeeze
import dev.kursor.ktensorflow.tensor.toFlatArray
import dev.kursor.ktensorflow.tensor.toLongTensor
import dev.kursor.ktensorflow.tensor.toUByteTensor
import dev.kursor.ktensorflow.tensor.transpose
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue

/**
 * Проверяет обещание 2.0: shape-трансформации не копируют память, а возвращают [TensorView],
 * который отображает координаты на исходный [PhysicalTensor]. Тесты пишут ЧЕРЕЗ view и
 * проверяют, что изменения видны в оригинале - это и есть отсутствие копирования.
 */
class TensorViewTest {

    private fun tensor2x3(): PhysicalTensor<Float> {
        val t = Tensor<Float>(TensorShape(2, 3))
        for (i in 0 until 2) for (j in 0 until 3) t[i, j] = (i * 3 + j).toFloat()
        return t
    }

    @Test
    fun reshapeReturnsAViewThatSharesMemoryWithTheOriginal() {
        val original = tensor2x3()
        val view = original.reshape(TensorShape(3, 2))

        assertTrue(view is TensorView<Float>, "reshape must not copy - expected a TensorView")
        assertEquals(3, view.shape.dimensions[0])
        assertEquals(2, view.shape.dimensions[1])

        // запись через view должна быть видна в оригинале
        view[0, 1] = 99f
        assertEquals(99f, original[0, 1])

        // и наоборот
        original[1, 2] = -7f
        assertEquals(-7f, view[2, 1])
    }

    @Test
    fun sliceReturnsAViewThatWritesThroughToTheOriginal() {
        val original = tensor2x3()
        val view = original.slice(arrayOf(0..1, 1..2))

        assertEquals(listOf(2, 2), view.shape.dimensions.toList())
        assertEquals(1f, view[0, 0])
        assertEquals(5f, view[1, 1])

        view[0, 0] = 42f
        assertEquals(42f, original[0, 1])
    }

    @Test
    fun transposeReturnsAViewThatWritesThroughToTheOriginal() {
        val original = tensor2x3()
        val view = original.transpose()

        assertEquals(listOf(3, 2), view.shape.dimensions.toList())
        assertEquals(original[1, 2], view[2, 1])

        view[2, 0] = 55f
        assertEquals(55f, original[0, 2])
    }

    @Test
    fun flattenReturnsAViewOverTheSameMemory() {
        val original = tensor2x3()
        val view = original.flatten()

        assertEquals(listOf(6), view.shape.dimensions.toList())
        view[4] = 13f
        assertEquals(13f, original[1, 1])
    }

    @Test
    fun squeezeDropsDimensionsOfSizeOne() {
        val original = Tensor<Float>(TensorShape(1, 3, 1))
        for (j in 0 until 3) original[0, j, 0] = j.toFloat()

        val squeezed = original.squeeze()

        assertEquals(listOf(3), squeezed.shape.dimensions.toList())
        assertEquals(2f, squeezed[2])

        squeezed[0] = 17f
        assertEquals(17f, original[0, 0, 0])
    }

    @Test
    fun permutedMapsViewAxesOntoTheOriginalAxes() {
        val original = Tensor<Float>(TensorShape(2, 3, 4))
        for (i in 0 until 2) for (j in 0 until 3) for (k in 0 until 4) {
            original[i, j, k] = (i * 100 + j * 10 + k).toFloat()
        }

        val view = original.permuted(2, 0, 1)

        assertEquals(listOf(4, 2, 3), view.shape.dimensions.toList())
        for (a in 0 until 4) for (b in 0 until 2) for (c in 0 until 3) {
            assertEquals(original[b, c, a], view[a, b, c], "mismatch at view[$a,$b,$c]")
        }
    }

    @Test
    fun toPhysicalMaterializesAnIndependentCopy() {
        val original = tensor2x3()
        val view = original.transpose()
        val copy = view.toPhysical()

        assertEquals(view[2, 1], copy[2, 1])

        // после материализации связь с оригиналом должна пропасть
        original[1, 2] = 123f
        assertEquals(123f, view[2, 1], "view must still track the original")
        assertNotEquals(123f, copy[2, 1], "toPhysical() must produce a detached copy")
    }

    @Test
    fun toFlatArrayReturnsElementsInRowMajorOrderForEveryPrimitiveType() {
        assertContentEquals(floatArrayOf(0f, 1f, 2f, 3f, 4f, 5f), tensor2x3().toFlatArray())

        val ints = Tensor<Int>(TensorShape(2, 2))
        for (i in 0 until 2) for (j in 0 until 2) ints[i, j] = i * 2 + j
        assertContentEquals(intArrayOf(0, 1, 2, 3), ints.toFlatArray())

        val longs = Tensor<Long>(TensorShape(3))
        for (i in 0 until 3) longs[i] = i.toLong()
        assertContentEquals(longArrayOf(0, 1, 2), longs.toFlatArray())
    }

    @Test
    fun toLongTensorAndToUByteTensorConvertElementTypes() {
        val source = tensor2x3()

        val asLong = source.toLongTensor()
        assertEquals(5L, asLong[1, 2])

        val asUByte = source.toUByteTensor()
        assertEquals(5.toUByte(), asUByte[1, 2])
    }

    @Test
    fun aViewOverAViewStillWritesThroughToTheOriginalMemory() {
        val original = tensor2x3()

        // reshape -> slice: цепочка из двух view поверх одного PhysicalTensor
        val chained = original.reshape(TensorShape(3, 2)).slice(arrayOf(1..2, 0..1))

        assertEquals(listOf(2, 2), chained.shape.dimensions.toList())
        chained[0, 0] = 77f
        assertEquals(77f, original[0, 2], "write through a chained view must reach the original")
    }
}
