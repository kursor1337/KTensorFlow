package tensor

import assertContentDeepEquals
import dev.kursor.ktensorflow.tensor.Tensor
import dev.kursor.ktensorflow.tensor.dec
import dev.kursor.ktensorflow.tensor.div
import dev.kursor.ktensorflow.tensor.inc
import dev.kursor.ktensorflow.tensor.minus
import dev.kursor.ktensorflow.tensor.plus
import dev.kursor.ktensorflow.tensor.rem
import dev.kursor.ktensorflow.tensor.times
import dev.kursor.ktensorflow.tensor.toArray
import dev.kursor.ktensorflow.tensor.unaryMinus
import dev.kursor.ktensorflow.tensor.unaryPlus
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
        val actual = (first + second).toPhysical().toArray<Array<Array<Array<FloatArray>>>>()
        assertContentDeepEquals(expected, actual)
    }

    @Test
    fun plusIntTest() {
        val first = Tensor<Int>(createDataInt(1))
        val second = Tensor<Int>(createDataInt(2))
        val expected = createDataInt(3)
        val actual = (first + second).toPhysical().toArray<Array<Array<Array<IntArray>>>>()
        assertContentDeepEquals(expected, actual)
    }

    @Test
    fun plusUByteTest() {
        val first = Tensor<UByte>(createDataUByte(1u))
        val second = Tensor<UByte>(createDataUByte(2u))
        val expected = createDataUByte(3u)
        val actual = (first + second).toPhysical().toArray<Array<Array<Array<UByteArray>>>>()
        assertContentDeepEquals(expected, actual)
    }

    @Test
    fun plusLongTest() {
        val first = Tensor<Long>(createDataLong(1))
        val second = Tensor<Long>(createDataLong(2))
        val expected = createDataLong(3)
        val actual = (first + second).toPhysical().toArray<Array<Array<Array<LongArray>>>>()
        assertContentDeepEquals(expected, actual)
    }

    @Test
    fun minusFloatTest() {
        val first = Tensor<Float>(createDataFloat(1f))
        val second = Tensor<Float>(createDataFloat(2f))
        val expected = createDataFloat(-1f)
        val actual = (first - second).toPhysical().toArray<Array<Array<Array<FloatArray>>>>()
        assertContentDeepEquals(expected, actual)
    }

    @Test
    fun minusIntTest() {
        val first = Tensor<Int>(createDataInt(1))
        val second = Tensor<Int>(createDataInt(2))
        val expected = createDataInt(-1)
        val actual = (first - second).toPhysical().toArray<Array<Array<Array<IntArray>>>>()
        assertContentDeepEquals(expected, actual)
    }

    @Test
    fun minusUByteTest() {
        val first = Tensor<UByte>(createDataUByte(2u))
        val second = Tensor<UByte>(createDataUByte(1u))
        val expected = createDataUByte(1u)
        val actual = (first - second).toPhysical().toArray<Array<Array<Array<UByteArray>>>>()
        assertContentDeepEquals(expected, actual)
    }

    @Test
    fun minusLongTest() {
        val first = Tensor<Long>(createDataLong(1))
        val second = Tensor<Long>(createDataLong(2))
        val expected = createDataLong(-1)
        val actual = (first - second).toPhysical().toArray<Array<Array<Array<LongArray>>>>()
        assertContentDeepEquals(expected, actual)
    }

    @Test
    fun timesFloatTest() {
        val first = Tensor<Float>(createDataFloat(3f))
        val second = Tensor<Float>(createDataFloat(2f))
        val expected = createDataFloat(6f)
        val actual = (first * second).toPhysical().toArray<Array<Array<Array<FloatArray>>>>()
        assertContentDeepEquals(expected, actual)
    }

    @Test
    fun timesIntTest() {
        val first = Tensor<Int>(createDataInt(3))
        val second = Tensor<Int>(createDataInt(2))
        val expected = createDataInt(6)
        val actual = (first * second).toPhysical().toArray<Array<Array<Array<IntArray>>>>()
        assertContentDeepEquals(expected, actual)
    }

    @Test
    fun timesUByteTest() {
        val first = Tensor<UByte>(createDataUByte(3u))
        val second = Tensor<UByte>(createDataUByte(2u))
        val expected = createDataUByte(6u)
        val actual = (first * second).toPhysical().toArray<Array<Array<Array<UByteArray>>>>()
        assertContentDeepEquals(expected, actual)
    }

    @Test
    fun timesLongTest() {
        val first = Tensor<Long>(createDataLong(3))
        val second = Tensor<Long>(createDataLong(2))
        val expected = createDataLong(6)
        val actual = (first * second).toPhysical().toArray<Array<Array<Array<LongArray>>>>()
        assertContentDeepEquals(expected, actual)
    }

    @Test
    fun divideFloatTest() {
        val first = Tensor<Float>(createDataFloat(6f))
        val second = Tensor<Float>(createDataFloat(3f))
        val expected = createDataFloat(2f)
        val actual = (first / second).toPhysical().toArray<Array<Array<Array<FloatArray>>>>()
        assertContentDeepEquals(expected, actual)
    }

    @Test
    fun divideIntTest() {
        val first = Tensor<Int>(createDataInt(6))
        val second = Tensor<Int>(createDataInt(2))
        val expected = createDataInt(3)
        val actual = (first / second).toPhysical().toArray<Array<Array<Array<IntArray>>>>()
        assertContentDeepEquals(expected, actual)
    }

    @Test
    fun divideUByteTest() {
        val first = Tensor<UByte>(createDataUByte(4u))
        val second = Tensor<UByte>(createDataUByte(2u))
        val expected = createDataUByte(2u)
        val actual = (first / second).toPhysical().toArray<Array<Array<Array<UByteArray>>>>()
        assertContentDeepEquals(expected, actual)
    }

    @Test
    fun divideLongTest() {
        val first = Tensor<Long>(createDataLong(4))
        val second = Tensor<Long>(createDataLong(2))
        val expected = createDataLong(2)
        val actual = (first / second).toPhysical().toArray<Array<Array<Array<LongArray>>>>()
        assertContentDeepEquals(expected, actual)
    }

    @Test
    fun remFloatTest() {
        val first = Tensor<Float>(createDataFloat(4f))
        val second = Tensor<Float>(createDataFloat(3f))
        val expected = createDataFloat(4f % 3)
        val actual = (first % second).toPhysical().toArray<Array<Array<Array<FloatArray>>>>()
        assertContentDeepEquals(expected, actual)
    }

    @Test
    fun remIntTest() {
        val first = Tensor<Int>(createDataInt(4))
        val second = Tensor<Int>(createDataInt(3))
        val expected = createDataInt(1)
        val actual = (first % second).toPhysical().toArray<Array<Array<Array<IntArray>>>>()
        assertContentDeepEquals(expected, actual)
    }

    @Test
    fun remUByteTest() {
        val first = Tensor<UByte>(createDataUByte(4u))
        val second = Tensor<UByte>(createDataUByte(3u))
        val expected = createDataUByte(1u)
        val actual = (first % second).toPhysical().toArray<Array<Array<Array<UByteArray>>>>()
        assertContentDeepEquals(expected, actual)
    }

    @Test
    fun remLongTest() {
        val first = Tensor<Long>(createDataLong(4))
        val second = Tensor<Long>(createDataLong(3))
        val expected = createDataLong(1)
        val actual = (first % second).toPhysical().toArray<Array<Array<Array<LongArray>>>>()
        assertContentDeepEquals(expected, actual)
    }

    // --- унарные операторы ---
    // Каждый объявлен отдельно для Float/Int/Long/UByte, поэтому перепутанный знак или
    // слагаемое в одной из перегрузок не виден, пока не проверена каждая из них.

    @Test
    fun incFloatTest() {
        val tensor = Tensor<Float>(createDataFloat(1f))
        val expected = createDataFloat(2f)
        val actual = tensor.inc().toPhysical().toArray<Array<Array<Array<FloatArray>>>>()
        assertContentDeepEquals(expected, actual)
    }

    @Test
    fun incIntTest() {
        val tensor = Tensor<Int>(createDataInt(1))
        val expected = createDataInt(2)
        val actual = tensor.inc().toPhysical().toArray<Array<Array<Array<IntArray>>>>()
        assertContentDeepEquals(expected, actual)
    }

    @Test
    fun incLongTest() {
        val tensor = Tensor<Long>(createDataLong(1))
        val expected = createDataLong(2)
        val actual = tensor.inc().toPhysical().toArray<Array<Array<Array<LongArray>>>>()
        assertContentDeepEquals(expected, actual)
    }

    @Test
    fun incUByteTest() {
        val tensor = Tensor<UByte>(createDataUByte(1u))
        val expected = createDataUByte(2u)
        val actual = tensor.inc().toPhysical().toArray<Array<Array<Array<UByteArray>>>>()
        assertContentDeepEquals(expected, actual)
    }

    @Test
    fun incUByteWrapsAroundAtTheTopOfTheRange() {
        // 255 + 1 не влезает в UByte и обязано дать 0, а не выбросить исключение
        val tensor = Tensor<UByte>(createDataUByte(255u))
        val expected = createDataUByte(0u)
        val actual = tensor.inc().toPhysical().toArray<Array<Array<Array<UByteArray>>>>()
        assertContentDeepEquals(expected, actual)
    }

    @Test
    fun decFloatTest() {
        val tensor = Tensor<Float>(createDataFloat(2f))
        val expected = createDataFloat(1f)
        val actual = tensor.dec().toPhysical().toArray<Array<Array<Array<FloatArray>>>>()
        assertContentDeepEquals(expected, actual)
    }

    @Test
    fun decIntTest() {
        val tensor = Tensor<Int>(createDataInt(2))
        val expected = createDataInt(1)
        val actual = tensor.dec().toPhysical().toArray<Array<Array<Array<IntArray>>>>()
        assertContentDeepEquals(expected, actual)
    }

    @Test
    fun decLongTest() {
        val tensor = Tensor<Long>(createDataLong(2))
        val expected = createDataLong(1)
        val actual = tensor.dec().toPhysical().toArray<Array<Array<Array<LongArray>>>>()
        assertContentDeepEquals(expected, actual)
    }

    @Test
    fun decUByteTest() {
        val tensor = Tensor<UByte>(createDataUByte(2u))
        val expected = createDataUByte(1u)
        val actual = tensor.dec().toPhysical().toArray<Array<Array<Array<UByteArray>>>>()
        assertContentDeepEquals(expected, actual)
    }

    @Test
    fun decUByteWrapsAroundAtTheBottomOfTheRange() {
        // 0 - 1 не влезает в UByte и обязано дать 255
        val tensor = Tensor<UByte>(createDataUByte(0u))
        val expected = createDataUByte(255u)
        val actual = tensor.dec().toPhysical().toArray<Array<Array<Array<UByteArray>>>>()
        assertContentDeepEquals(expected, actual)
    }

    @Test
    fun unaryMinusFloatTest() {
        val tensor = Tensor<Float>(createDataFloat(3f))
        val expected = createDataFloat(-3f)
        val actual = (-tensor).toPhysical().toArray<Array<Array<Array<FloatArray>>>>()
        assertContentDeepEquals(expected, actual)
    }

    @Test
    fun unaryMinusIntTest() {
        val tensor = Tensor<Int>(createDataInt(3))
        val expected = createDataInt(-3)
        val actual = (-tensor).toPhysical().toArray<Array<Array<Array<IntArray>>>>()
        assertContentDeepEquals(expected, actual)
    }

    @Test
    fun unaryMinusLongTest() {
        val tensor = Tensor<Long>(createDataLong(3))
        val expected = createDataLong(-3)
        val actual = (-tensor).toPhysical().toArray<Array<Array<Array<LongArray>>>>()
        assertContentDeepEquals(expected, actual)
    }

    @Test
    fun unaryMinusAppliedTwiceRestoresTheOriginal() {
        val tensor = Tensor<Float>(createDataFloat(3f))
        val expected = createDataFloat(3f)
        val actual = (-(-tensor)).toPhysical().toArray<Array<Array<Array<FloatArray>>>>()
        assertContentDeepEquals(expected, actual)
    }

    @Test
    fun unaryPlusFloatTest() {
        val tensor = Tensor<Float>(createDataFloat(-3f))
        val expected = createDataFloat(-3f)
        val actual = (+tensor).toPhysical().toArray<Array<Array<Array<FloatArray>>>>()
        assertContentDeepEquals(expected, actual)
    }

    @Test
    fun unaryPlusIntTest() {
        val tensor = Tensor<Int>(createDataInt(-3))
        val expected = createDataInt(-3)
        val actual = (+tensor).toPhysical().toArray<Array<Array<Array<IntArray>>>>()
        assertContentDeepEquals(expected, actual)
    }

    @Test
    fun unaryPlusLongTest() {
        val tensor = Tensor<Long>(createDataLong(-3))
        val expected = createDataLong(-3)
        val actual = (+tensor).toPhysical().toArray<Array<Array<Array<LongArray>>>>()
        assertContentDeepEquals(expected, actual)
    }
}
