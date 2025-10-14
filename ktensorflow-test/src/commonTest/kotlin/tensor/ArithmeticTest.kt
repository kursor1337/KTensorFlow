package tensor

import assertContentDeepEquals
import dev.kursor.ktensorflow.tensor.Tensor
import dev.kursor.ktensorflow.tensor.div
import dev.kursor.ktensorflow.tensor.minus
import dev.kursor.ktensorflow.tensor.plus
import dev.kursor.ktensorflow.tensor.rem
import dev.kursor.ktensorflow.tensor.times
import dev.kursor.ktensorflow.tensor.toArray
import kotlin.test.Test

@OptIn(ExperimentalUnsignedTypes::class)
class ArithmeticTest {

    fun createDataFloat(number: Float): Array<Array<Array<FloatArray>>> {
        return Array(3) { i ->
            Array(5) { j ->
                Array(7) { k ->
                    FloatArray(11) {
                        number
                    }
                }
            }
        }
    }

    fun createDataInt(number: Int): Array<Array<Array<IntArray>>> {
        return Array(3) { i ->
            Array(5) { j ->
                Array(7) { k ->
                    IntArray(11) {
                        number
                    }
                }
            }
        }
    }

    fun createDataUByte(number: UByte): Array<Array<Array<UByteArray>>> {
        return Array(3) { i ->
            Array(5) { j ->
                Array(7) { k ->
                    UByteArray(11) {
                        number
                    }
                }
            }
        }
    }

    fun createDataLong(number: Long): Array<Array<Array<LongArray>>> {
        return Array(3) { i ->
            Array(5) { j ->
                Array(7) { k ->
                    LongArray(1) {
                        number
                    }
                }
            }
        }
    }

    @Test
    fun plusFloatTest() {
        val first = Tensor<Float>(createDataFloat(1f))
        val second = Tensor<Float>(createDataFloat(2f))
        val expected = createDataFloat(3f)
        val actual = (first + second).toArray<Array<Array<Array<FloatArray>>>>()
        assertContentDeepEquals(expected, actual)
    }

    @Test
    fun plusIntTest() {
        val first = Tensor<Int>(createDataInt(1))
        val second = Tensor<Int>(createDataInt(2))
        val expected = createDataInt(3)
        val actual = (first + second).toArray<Array<Array<Array<IntArray>>>>()
        assertContentDeepEquals(expected, actual)
    }

    @Test
    fun plusUByteTest() {
        val first = Tensor<UByte>(createDataUByte(1u))
        val second = Tensor<UByte>(createDataUByte(2u))
        val expected = createDataUByte(3u)
        val actual = (first + second).toArray<Array<Array<Array<UByteArray>>>>()
        assertContentDeepEquals(expected, actual)
    }

    @Test
    fun plusLongTest() {
        val first = Tensor<Long>(createDataLong(1))
        val second = Tensor<Long>(createDataLong(2))
        val expected = createDataLong(3)
        val actual = (first + second).toArray<Array<Array<Array<LongArray>>>>()
        assertContentDeepEquals(expected, actual)
    }

    @Test
    fun minusFloatTest() {
        val first = Tensor<Float>(createDataFloat(1f))
        val second = Tensor<Float>(createDataFloat(2f))
        val expected = createDataFloat(-1f)
        val actual = (first - second).toArray<Array<Array<Array<FloatArray>>>>()
        assertContentDeepEquals(expected, actual)
    }

    @Test
    fun minusIntTest() {
        val first = Tensor<Int>(createDataInt(1))
        val second = Tensor<Int>(createDataInt(2))
        val expected = createDataInt(-1)
        val actual = (first - second).toArray<Array<Array<Array<IntArray>>>>()
        assertContentDeepEquals(expected, actual)
    }

    @Test
    fun minusUByteTest() {
        val first = Tensor<UByte>(createDataUByte(2u))
        val second = Tensor<UByte>(createDataUByte(1u))
        val expected = createDataUByte(1u)
        val actual = (first - second).toArray<Array<Array<Array<UByteArray>>>>()
        assertContentDeepEquals(expected, actual)
    }

    @Test
    fun minusLongTest() {
        val first = Tensor<Long>(createDataLong(1))
        val second = Tensor<Long>(createDataLong(2))
        val expected = createDataLong(-1)
        val actual = (first - second).toArray<Array<Array<Array<LongArray>>>>()
        assertContentDeepEquals(expected, actual)
    }

    @Test
    fun timesFloatTest() {
        val first = Tensor<Float>(createDataFloat(3f))
        val second = Tensor<Float>(createDataFloat(2f))
        val expected = createDataFloat(6f)
        val actual = (first * second).toArray<Array<Array<Array<FloatArray>>>>()
        assertContentDeepEquals(expected, actual)
    }

    @Test
    fun timesIntTest() {
        val first = Tensor<Int>(createDataInt(3))
        val second = Tensor<Int>(createDataInt(2))
        val expected = createDataInt(6)
        val actual = (first * second).toArray<Array<Array<Array<IntArray>>>>()
        assertContentDeepEquals(expected, actual)
    }

    @Test
    fun timesUByteTest() {
        val first = Tensor<UByte>(createDataUByte(3u))
        val second = Tensor<UByte>(createDataUByte(2u))
        val expected = createDataUByte(6u)
        val actual = (first * second).toArray<Array<Array<Array<UByteArray>>>>()
        assertContentDeepEquals(expected, actual)
    }

    @Test
    fun timesLongTest() {
        val first = Tensor<Long>(createDataLong(3))
        val second = Tensor<Long>(createDataLong(2))
        val expected = createDataLong(6)
        val actual = (first * second).toArray<Array<Array<Array<LongArray>>>>()
        assertContentDeepEquals(expected, actual)
    }

    @Test
    fun divideFloatTest() {
        val first = Tensor<Float>(createDataFloat(6f))
        val second = Tensor<Float>(createDataFloat(3f))
        val expected = createDataFloat(2f)
        val actual = (first / second).toArray<Array<Array<Array<FloatArray>>>>()
        assertContentDeepEquals(expected, actual)
    }

    @Test
    fun divideIntTest() {
        val first = Tensor<Int>(createDataInt(6))
        val second = Tensor<Int>(createDataInt(2))
        val expected = createDataInt(3)
        val actual = (first / second).toArray<Array<Array<Array<IntArray>>>>()
        assertContentDeepEquals(expected, actual)
    }

    @Test
    fun divideUByteTest() {
        val first = Tensor<UByte>(createDataUByte(4u))
        val second = Tensor<UByte>(createDataUByte(2u))
        val expected = createDataUByte(2u)
        val actual = (first / second).toArray<Array<Array<Array<UByteArray>>>>()
        assertContentDeepEquals(expected, actual)
    }

    @Test
    fun divideLongTest() {
        val first = Tensor<Long>(createDataLong(4))
        val second = Tensor<Long>(createDataLong(2))
        val expected = createDataLong(2)
        val actual = (first / second).toArray<Array<Array<Array<LongArray>>>>()
        assertContentDeepEquals(expected, actual)
    }

    @Test
    fun remFloatTest() {
        val first = Tensor<Float>(createDataFloat(4f))
        val second = Tensor<Float>(createDataFloat(3f))
        val expected = createDataFloat(4f % 3)
        val actual = (first % second).toArray<Array<Array<Array<FloatArray>>>>()
        assertContentDeepEquals(expected, actual)
    }

    @Test
    fun remIntTest() {
        val first = Tensor<Int>(createDataInt(4))
        val second = Tensor<Int>(createDataInt(3))
        val expected = createDataInt(1)
        val actual = (first % second).toArray<Array<Array<Array<IntArray>>>>()
        assertContentDeepEquals(expected, actual)
    }

    @Test
    fun remUByteTest() {
        val first = Tensor<UByte>(createDataUByte(4u))
        val second = Tensor<UByte>(createDataUByte(3u))
        val expected = createDataUByte(1u)
        val actual = (first % second).toArray<Array<Array<Array<UByteArray>>>>()
        assertContentDeepEquals(expected, actual)
    }

    @Test
    fun remLongTest() {
        val first = Tensor<Long>(createDataLong(4))
        val second = Tensor<Long>(createDataLong(3))
        val expected = createDataLong(1)
        val actual = (first % second).toArray<Array<Array<Array<LongArray>>>>()
        assertContentDeepEquals(expected, actual)
    }
}