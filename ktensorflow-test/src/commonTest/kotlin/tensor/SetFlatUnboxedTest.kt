package tensor

import dev.kursor.ktensorflow.tensor.PhysicalTensor
import dev.kursor.ktensorflow.tensor.Tensor
import dev.kursor.ktensorflow.tensor.TensorDataType
import dev.kursor.ktensorflow.tensor.TensorShape
import dev.kursor.ktensorflow.tensor.setFlatUnboxed
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

/**
 * setFlatUnboxed пишет прямо в байты тензора, минуя обобщённый setFlat. Раскладка байтов обязана
 * совпадать с setFlat бит в бит: эти байты уходят в интерпретатор как есть.
 */
class SetFlatUnboxedTest {

    private val shape = TensorShape(2, 3)

    private fun <T : Any> sameBytes(
        dataType: TensorDataType<T>,
        values: List<T>,
        setUnboxed: PhysicalTensor<T>.(Int, T) -> Unit
    ) {
        val generic = Tensor(dataType, shape)
        val unboxed = Tensor(dataType, shape)

        values.forEachIndexed { index, value ->
            generic.setFlat(index, value)
            unboxed.setUnboxed(index, value)
        }

        assertContentEquals(generic.data, unboxed.data, "$dataType")
        values.forEachIndexed { index, value -> assertEquals(value, unboxed.getFlat(index), "$dataType at $index") }
    }

    @Test
    fun writesTheSameBytesAsSetFlatForEveryDataType() {
        sameBytes(TensorDataType.Float32, listOf(0f, -1.5f, Float.MAX_VALUE, Float.MIN_VALUE, Float.NaN, -0f)) { i, v ->
            setFlatUnboxed(i, v)
        }
        sameBytes(TensorDataType.Int32, listOf(0, -1, Int.MAX_VALUE, Int.MIN_VALUE, 0x01020304, 255)) { i, v ->
            setFlatUnboxed(i, v)
        }
        sameBytes(TensorDataType.Int64, listOf(0L, -1L, Long.MAX_VALUE, Long.MIN_VALUE, 0x0102030405060708L, 1L shl 40)) { i, v ->
            setFlatUnboxed(i, v)
        }
        sameBytes(TensorDataType.UInt8, listOf(0u, 1u, 127u, 128u, 254u, 255u).map { it.toUByte() }) { i, v ->
            setFlatUnboxed(i, v)
        }
    }

    @Test
    fun rejectsAnIndexOutsideTheTensor() {
        val tensor = Tensor(TensorDataType.Float32, shape)

        assertFailsWith<IndexOutOfBoundsException> { tensor.setFlatUnboxed(shape.flatSize, 1f) }
        assertFailsWith<IndexOutOfBoundsException> { tensor.setFlatUnboxed(-1, 1f) }
    }
}
