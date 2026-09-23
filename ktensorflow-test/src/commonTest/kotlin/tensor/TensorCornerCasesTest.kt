package tensor

import dev.kursor.ktensorflow.tensor.Tensor
import dev.kursor.ktensorflow.tensor.TensorShape
import dev.kursor.ktensorflow.tensor.argmax
import dev.kursor.ktensorflow.tensor.avg
import dev.kursor.ktensorflow.tensor.flatten
import dev.kursor.ktensorflow.tensor.max
import dev.kursor.ktensorflow.tensor.min
import dev.kursor.ktensorflow.tensor.normalize
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
}
