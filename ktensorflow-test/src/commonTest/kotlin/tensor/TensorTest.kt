package tensor

import assertContentDeepEquals
import dev.kursor.ktensorflow.tensor.Tensor
import dev.kursor.ktensorflow.tensor.toArray
import kotlin.test.Test

class TensorTest {
    @Test
    fun reshapeSmallFloat() {
        val data = Array(2) { i -> FloatArray(3) { j -> (i * 31 + j).toFloat() } }
        val tensor = Tensor<Float>(data)
        val reshapedBack = tensor.toArray<Array<FloatArray>>()
        assertContentDeepEquals(data, reshapedBack)
    }

    @Test
    fun reshapeLargeFloat() {
        val data = Array(8) { i ->
            Array(8) { j ->
                FloatArray(12) { k ->
                    (i * 31 + j * 31 + k).toFloat()
                }
            }
        }

        val tensor = Tensor<Float>(data)
        val reshapedBack = tensor.toArray<Array<Array<FloatArray>>>()
        assertContentDeepEquals(data, reshapedBack)
    }

    @Test
    fun reshapeVeryLargeFloat() {
        val data = Array(100) { i ->
            Array(8) { j ->
                Array(99) { k ->
                    FloatArray(73) { l ->
                        (i * 31 + j * 31 + k * 31 + l).toFloat()
                    }
                }
            }
        }

        val tensor = Tensor<Float>(data)
        val reshapedBack = tensor.toArray<Array<Array<Array<FloatArray>>>>()
        assertContentDeepEquals(data, reshapedBack)
    }

    @Test
    fun reshapeSmallInt() {
        val data = Array(2) { i -> IntArray(3) { j -> i * 31 + j } }
        val tensor = Tensor<Int>(data)
        val reshapedBack = tensor.toArray<Array<IntArray>>()
        assertContentDeepEquals(data, reshapedBack)
    }

    @Test
    fun reshapeLargeInt() {
        val data = Array(8) { i ->
            Array(8) { j ->
                IntArray(12) { k ->
                    i * 31 + j * 31 + k
                }
            }
        }

        val tensor = Tensor<Int>(data)
        val reshapedBack = tensor.toArray<Array<Array<IntArray>>>()
        assertContentDeepEquals(data, reshapedBack)
    }

    @Test
    fun reshapeVeryLargeInt() {
        val data = Array(100) { i ->
            Array(8) { j ->
                Array(99) { k ->
                    IntArray(73) { l ->
                        i * 31 + j * 31 + k * 31 + l
                    }
                }
            }
        }

        val tensor = Tensor<Int>(data)
        val reshapedBack = tensor.toArray<Array<Array<Array<IntArray>>>>()
        assertContentDeepEquals(data, reshapedBack)
    }

    @OptIn(ExperimentalUnsignedTypes::class)
    @Test
    fun reshapeSmallByte() {
        val data = Array(2) { i -> UByteArray(3) { j -> (i * 31 + j).toUByte() } }
        val tensor = Tensor<UByte>(data)
        val reshapedBack = tensor.toArray<Array<UByteArray>>()
        assertContentDeepEquals(data, reshapedBack)
    }

    @OptIn(ExperimentalUnsignedTypes::class)
    @Test
    fun reshapeLargeByte() {
        val data = Array(8) { i ->
            Array(8) { j ->
                UByteArray(12) { k ->
                    (i * 31 + j * 31 + k).toUByte()
                }
            }
        }

        val tensor = Tensor<UByte>(data)
        val reshapedBack = tensor.toArray<Array<Array<UByteArray>>>()
        assertContentDeepEquals(data, reshapedBack)
    }

    @OptIn(ExperimentalUnsignedTypes::class)
    @Test
    fun reshapeVeryLargeByte() {
        val data = Array(100) { i ->
            Array(8) { j ->
                Array(99) { k ->
                    UByteArray(73) { l ->
                        (i * 31 + j * 31 + k * 31 + l).toUByte()
                    }
                }
            }
        }

        val tensor = Tensor<UByte>(data)
        val reshapedBack = tensor.toArray<Array<Array<Array<UByteArray>>>>()
        assertContentDeepEquals(data, reshapedBack)
    }

    @Test
    fun reshapeSmallLong() {
        val data = Array(2) { i -> LongArray(3) { j -> (i * 31 + j).toLong() } }
        val tensor = Tensor<Long>(data)
        val reshapedBack = tensor.toArray<Array<LongArray>>()
        assertContentDeepEquals(data, reshapedBack)
    }

    @Test
    fun reshapeLargeLong() {
        val data = Array(8) { i ->
            Array(8) { j ->
                LongArray(12) { k ->
                    (i * 31 + j * 31 + k).toLong()
                }
            }
        }

        val tensor = Tensor<Long>(data)
        val reshapedBack = tensor.toArray<Array<Array<LongArray>>>()
        assertContentDeepEquals(data, reshapedBack)
    }

    @Test
    fun reshapeVeryLargeLong() {
        val data = Array(100) { i ->
            Array(8) { j ->
                Array(99) { k ->
                    LongArray(73) { l ->
                        (i * 31 + j * 31 + k * 31 + l).toLong()
                    }
                }
            }
        }

        val tensor = Tensor<Long>(data)
        val reshapedBack = tensor.toArray<Array<Array<Array<LongArray>>>>()
        assertContentDeepEquals(data, reshapedBack)
    }
}