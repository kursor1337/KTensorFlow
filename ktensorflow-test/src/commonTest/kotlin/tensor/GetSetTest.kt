package tensor

import dev.kursor.ktensorflow.tensor.Tensor
import dev.kursor.ktensorflow.tensor.get
import dev.kursor.ktensorflow.tensor.set
import kotlin.test.Test
import kotlin.test.assertEquals

class GetSetTest {

    @Test
    fun getTest() {
        val data = Array(2) { i -> FloatArray(3) { j -> (i * 3 + j).toFloat() } }
        val tensor = Tensor<Float>(data)
        assertEquals(5f, tensor[1, 2])
    }

    @Test
    fun setTest() {
        val data = Array(2) { i -> FloatArray(3) { j -> (i * 3 + j).toFloat() } }
        val tensor = Tensor<Float>(data)
        tensor[1, 2] = 10f
        assertEquals(10f, tensor[1, 2])
    }
}