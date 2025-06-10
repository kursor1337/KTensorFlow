import dev.kursor.ktensorflow.Tensor
import dev.kursor.ktensorflow.typedData
import kotlin.test.Test

class TensorTest {
    @Test
    fun reshapeSmallFloat() {
        val data = Array(2) { i -> FloatArray(3) { j -> (i * 31 + j).toFloat() } }
        val tensor = Tensor(data)
        val reshapedBack = tensor.typedData<Array<FloatArray>>()
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

        val tensor = Tensor(data)
        val reshapedBack = tensor.typedData<Array<Array<FloatArray>>>()
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

        val tensor = Tensor(data)
        val reshapedBack = tensor.typedData<Array<Array<Array<FloatArray>>>>()
        assertContentDeepEquals(data, reshapedBack)
    }

    @Test
    fun reshapeSmallInt() {
        val data = Array(2) { i -> IntArray(3) { j -> i * 31 + j } }
        val tensor = Tensor(data)
        val reshapedBack = tensor.typedData<Array<IntArray>>()
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

        val tensor = Tensor(data)
        val reshapedBack = tensor.typedData<Array<Array<IntArray>>>()
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

        val tensor = Tensor(data)
        val reshapedBack = tensor.typedData<Array<Array<Array<IntArray>>>>()
        assertContentDeepEquals(data, reshapedBack)
    }

    @Test
    fun reshapeSmallByte() {
        val data = Array(2) { i -> ByteArray(3) { j -> (i * 31 + j).toByte() } }
        val tensor = Tensor(data)
        val reshapedBack = tensor.typedData<Array<ByteArray>>()
        assertContentDeepEquals(data, reshapedBack)
    }

    @Test
    fun reshapeLargeByte() {
        val data = Array(8) { i ->
            Array(8) { j ->
                ByteArray(12) { k ->
                    (i * 31 + j * 31 + k).toByte()
                }
            }
        }

        val tensor = Tensor(data)
        val reshapedBack = tensor.typedData<Array<Array<ByteArray>>>()
        assertContentDeepEquals(data, reshapedBack)
    }

    @Test
    fun reshapeVeryLargeByte() {
        val data = Array(100) { i ->
            Array(8) { j ->
                Array(99) { k ->
                    ByteArray(73) { l ->
                        (i * 31 + j * 31 + k * 31 + l).toByte()
                    }
                }
            }
        }

        val tensor = Tensor(data)
        val reshapedBack = tensor.typedData<Array<Array<Array<ByteArray>>>>()
        assertContentDeepEquals(data, reshapedBack)
    }

    @Test
    fun reshapeSmallLong() {
        val data = Array(2) { i -> LongArray(3) { j -> (i * 31 + j).toLong() } }
        val tensor = Tensor(data)
        val reshapedBack = tensor.typedData<Array<LongArray>>()
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

        val tensor = Tensor(data)
        val reshapedBack = tensor.typedData<Array<Array<LongArray>>>()
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

        val tensor = Tensor(data)
        val reshapedBack = tensor.typedData<Array<Array<Array<LongArray>>>>()
        assertContentDeepEquals(data, reshapedBack)
    }
}