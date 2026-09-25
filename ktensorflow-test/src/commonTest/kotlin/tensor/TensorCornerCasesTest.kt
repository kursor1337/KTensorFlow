package tensor

import dev.kursor.ktensorflow.tensor.Tensor
import dev.kursor.ktensorflow.tensor.TensorShape
import dev.kursor.ktensorflow.tensor.argmax
import dev.kursor.ktensorflow.tensor.avg
import dev.kursor.ktensorflow.tensor.flatten
import dev.kursor.ktensorflow.tensor.max
import dev.kursor.ktensorflow.tensor.min
import dev.kursor.ktensorflow.tensor.normalize
import dev.kursor.ktensorflow.tensor.permuted
import dev.kursor.ktensorflow.tensor.plus
import dev.kursor.ktensorflow.tensor.reshape
import dev.kursor.ktensorflow.tensor.slice
import dev.kursor.ktensorflow.tensor.squeeze
import dev.kursor.ktensorflow.tensor.sum
import dev.kursor.ktensorflow.tensor.toArray
import dev.kursor.ktensorflow.tensor.transpose
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFails
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Границы операций над тензором: вырожденные формы, одинаковые значения и заведомо неверные
 * аргументы. Остальные тесты модуля проверяют обычные случаи, а ломается такой код именно
 * здесь - причём молча, превращая данные в NaN, а не падая.
 */
class TensorCornerCasesTest {

    // --- вырожденные формы ---

    @Test
    fun scalarShapeHasASingleElement() {
        // Форма без измерений - это скаляр, в нём один элемент. Пустое произведение равно
        // единице, и flatSize обязан это выдержать, иначе squeeze() ниже падает.
        assertEquals(1, TensorShape().flatSize)
    }

    @Test
    fun squeezeOfATensorWhereEveryDimensionIsOneKeepsTheValue() {
        val tensor = Tensor<Float>(shape = TensorShape(1, 1, 1))
        tensor.setFlat(0, 42f)

        val squeezed = tensor.squeeze()

        assertEquals(42f, squeezed.getFlat(0))
    }

    @Test
    fun squeezeKeepsDimensionsLargerThanOne() {
        val tensor = Tensor<Float>(shape = TensorShape(1, 3, 1))

        val squeezed = tensor.squeeze()

        assertEquals(listOf(3), squeezed.shape.dimensions.toList())
    }

    @Test
    fun aSingleElementTensorSurvivesFlattenAndReshape() {
        val tensor = Tensor<Float>(shape = TensorShape(1))
        tensor.setFlat(0, 7f)

        assertEquals(7f, tensor.flatten().getFlat(0))
        assertEquals(7f, tensor.reshape(TensorShape(1, 1)).getFlat(0))
    }

    // --- normalize ---

    @Test
    fun normalizeOfAConstantTensorDoesNotProduceNaN() {
        // Сплошная заливка - совершенно обычный вход (однотонный кадр), и min == max.
        // Деление на ноль превратило бы весь тензор в NaN, который потом молча
        // расползается по инференсу.
        val tensor = Tensor<Float>(shape = TensorShape(2, 2))
        repeat(4) { tensor.setFlat(it, 5f) }

        val normalized = tensor.normalize()

        repeat(4) { i ->
            val value = normalized.getFlat(i)
            assertFalse(value.isNaN(), "element $i must not be NaN")
            assertEquals(0f, value, "a constant tensor normalizes to zeros")
        }
    }

    @Test
    fun normalizeMapsTheRangeOntoZeroToOne() {
        val tensor = Tensor<Float>(shape = TensorShape(3))
        tensor.setFlat(0, -2f)
        tensor.setFlat(1, 0f)
        tensor.setFlat(2, 2f)

        val normalized = tensor.normalize()

        assertEquals(0f, normalized.getFlat(0))
        assertEquals(0.5f, normalized.getFlat(1))
        assertEquals(1f, normalized.getFlat(2))
    }

    @Test
    fun normalizeOfASingleElementTensorDoesNotProduceNaN() {
        val tensor = Tensor<Float>(shape = TensorShape(1))
        tensor.setFlat(0, 3f)

        assertFalse(tensor.normalize().getFlat(0).isNaN())
    }

    // --- агрегаты на вырожденных данных ---

    @Test
    fun aggregatesOverASingleElementReturnThatElement() {
        val tensor = Tensor<Float>(shape = TensorShape(1))
        tensor.setFlat(0, -4f)

        assertEquals(-4f, tensor.sum())
        assertEquals(-4f, tensor.avg())
        assertEquals(-4f, tensor.min())
        assertEquals(-4f, tensor.max())
        assertEquals(listOf(0), tensor.argmax().toList())
    }

    @Test
    fun argmaxReturnsTheFirstIndexWhenValuesAreTied() {
        val tensor = Tensor<Float>(shape = TensorShape(4))
        tensor.setFlat(0, 1f)
        tensor.setFlat(1, 9f)
        tensor.setFlat(2, 9f)
        tensor.setFlat(3, 3f)

        assertEquals(listOf(1), tensor.argmax().toList())
    }

    // --- заведомо неверные аргументы ---

    @Test
    fun reshapeToAnIncompatibleSizeFails() {
        val tensor = Tensor<Float>(shape = TensorShape(2, 3))

        assertFails { tensor.reshape(TensorShape(4, 2)) }
    }

    @Test
    fun sliceOutsideTheTensorFails() {
        val tensor = Tensor<Float>(shape = TensorShape(2, 3))

        assertFails { tensor.slice(arrayOf(0..5, 0..2)).toPhysical() }
    }

    @Test
    fun sliceWithTheWrongNumberOfRangesFails() {
        val tensor = Tensor<Float>(shape = TensorShape(2, 3))

        assertFails { tensor.slice(arrayOf(0..1)) }
    }

    @Test
    fun addingTensorsOfDifferentShapesFails() {
        val a = Tensor<Float>(shape = TensorShape(2, 3))
        val b = Tensor<Float>(shape = TensorShape(3, 2))

        assertFails { (a + b).toPhysical().toArray<Array<FloatArray>>() }
    }

    @Test
    fun sliceOutsideAnAxisIsRejectedWhenTheViewIsCreated() {
        // (2, 3) = [[0,1,2],[3,4,5]]: столбцов 3 и 4 нет. Раньше view молча читал 3 и 4
        // из следующей строки и не падал, потому что плоские смещения оставались в массиве.
        val tensor = Tensor<Float>(shape = TensorShape(2, 3))
        repeat(6) { tensor.setFlat(it, it.toFloat()) }

        assertFails { tensor.slice(arrayOf(0..0, 0..4)) }
        assertFails { tensor.slice(arrayOf(-1..0, 0..2)) }
    }

    @Test
    fun anEmptySliceIsAllowed() {
        val tensor = Tensor<Float>(shape = TensorShape(2, 3))

        val empty = tensor.slice(arrayOf(1..0, 0..2))

        assertEquals(0, empty.shape.flatSize)
    }

    @Test
    fun permutedRejectsAxesThatAreNotAPermutation() {
        // На квадратном тензоре повтор оси не выводит смещения за массив, и раньше
        // permuted(1, 1) молча возвращал мусор [0,1,2,1,2,3,2,3,4]
        val tensor = Tensor<Float>(shape = TensorShape(3, 3))

        assertFails { tensor.permuted(1, 1) }
        assertFails { tensor.permuted(0, 2) }
        assertEquals(listOf(3, 3), tensor.permuted(1, 0).shape.dimensions.toList())
    }

    @Test
    fun negativeDimensionIsRejected() {
        assertFails { TensorShape(2, -1) }
    }

    @Test
    fun squeezeKeepsAZeroSizedDimension() {
        // Раньше squeeze выбрасывал и нулевые измерения, и пустой тензор (0, 3) превращался в (3)
        val tensor = Tensor<Float>(shape = TensorShape(1, 0, 3))

        assertEquals(listOf(0, 3), tensor.squeeze().shape.dimensions.toList())
    }

    // --- view поверх вырожденной формы ---

    @Test
    fun transposeOfASingleRowKeepsEveryValue() {
        val tensor = Tensor<Float>(shape = TensorShape(1, 3))
        repeat(3) { tensor.setFlat(it, it.toFloat()) }

        val transposed = tensor.transpose()

        assertEquals(listOf(3, 1), transposed.shape.dimensions.toList())
        repeat(3) { i ->
            assertEquals(i.toFloat(), transposed[intArrayOf(i, 0)], "element $i")
        }
    }

    @Test
    fun sliceOfASingleElementStillWritesThrough() {
        val tensor = Tensor<Float>(shape = TensorShape(3, 3))

        val cell = tensor.slice(arrayOf(1..1, 1..1))
        cell.setFlat(0, 8f)

        assertEquals(8f, tensor[intArrayOf(1, 1)])
        assertTrue(cell.shape.flatSize == 1)
    }

    // --- toArray ---

    @OptIn(ExperimentalUnsignedTypes::class)
    @Test
    fun oneDimensionalUByteTensorConvertsToAnArray() {
        // Выход квантованной модели формы [N]: на Android падал с "Bad number of dimensions: 0",
        // потому что UByteArray собирается отдельно, а других измерений не оставалось
        val tensor = Tensor<UByte>(shape = TensorShape(3))
        tensor.setFlat(2, 7.toUByte())

        val array = tensor.toPhysical().toArray<UByteArray>()

        assertEquals(listOf<UByte>(0u, 0u, 7u), array.toList())
    }

    @Test
    fun scalarConvertsToAOneElementArray() {
        // Скаляр раньше падал на обеих платформах, причём с разными исключениями
        val scalar = Tensor<Float>(shape = TensorShape())
        scalar.setFlat(0, 3f)

        val array = scalar.toPhysical().toArray<FloatArray>()

        assertEquals(listOf(3f), array.toList())
    }

    @Test
    fun fiveDimensionalTensorConvertsToNestedArrays() {
        val tensor = Tensor<Float>(shape = TensorShape(1, 2, 1, 2, 3))
        repeat(12) { tensor.setFlat(it, it.toFloat()) }

        val array = tensor.toPhysical().toArray<Array<Array<Array<Array<FloatArray>>>>>()

        assertEquals(listOf(9f, 10f, 11f), array[0][1][0][1].toList())
    }

    // --- views: плоский доступ совпадает с вложенным ---

    @Test
    fun viewOfAViewKeepsEveryElementInPlace() {
        // (2, 3, 4) -> срез (1, 2, 3) -> перестановка (3, 1, 2): плоский доступ, вложенный доступ
        // и toPhysical обязаны давать одно и то же
        val tensor = Tensor<Float>(shape = TensorShape(2, 3, 4))
        repeat(24) { tensor.setFlat(it, it.toFloat()) }

        val view = tensor.slice(arrayOf(1..1, 1..2, 1..3)).permuted(2, 0, 1)
        val physical = view.toPhysical()

        assertEquals(listOf(3, 1, 2), view.shape.dimensions.toList())
        // Элемент view [c, n, h] = исходный [1 + n, 1 + h, 1 + c] = 12 * (1 + n) + 4 * (1 + h) + 1 + c
        val expected = mutableListOf<Float>()
        for (c in 0 until 3) for (n in 0 until 1) for (h in 0 until 2) {
            expected += (12 * (1 + n) + 4 * (1 + h) + 1 + c).toFloat()
        }
        assertEquals(expected, List(6) { view.getFlat(it) })
        assertEquals(expected, List(6) { physical.getFlat(it) })
        assertEquals(expected[5], view[intArrayOf(2, 0, 1)])
    }

    @Test
    fun flatAccessOutsideAViewFails() {
        val view = Tensor<Float>(shape = TensorShape(3, 3)).slice(arrayOf(0..1, 0..1))

        assertFails { view.getFlat(4) }
        assertFails { view.getFlat(-1) }
    }

    @Test
    fun writingThroughAPermutedViewChangesTheOriginal() {
        val tensor = Tensor<Float>(shape = TensorShape(2, 3))
        val transposed = tensor.permuted(1, 0)

        transposed.setFlat(1, 5f) // [0, 1] у view - это [1, 0] у исходника

        assertEquals(5f, tensor[intArrayOf(1, 0)])
    }
}
