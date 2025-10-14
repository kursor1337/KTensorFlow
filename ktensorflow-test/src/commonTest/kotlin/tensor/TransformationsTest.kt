package tensor

import assertContentDeepEquals
import dev.kursor.ktensorflow.tensor.Tensor
import dev.kursor.ktensorflow.tensor.TensorShape
import dev.kursor.ktensorflow.tensor.avg
import dev.kursor.ktensorflow.tensor.flatten
import dev.kursor.ktensorflow.tensor.forEach
import dev.kursor.ktensorflow.tensor.forEachIndexed
import dev.kursor.ktensorflow.tensor.map
import dev.kursor.ktensorflow.tensor.mapInPlace
import dev.kursor.ktensorflow.tensor.mapInPlaceIndexed
import dev.kursor.ktensorflow.tensor.mapIndexed
import dev.kursor.ktensorflow.tensor.reshape
import dev.kursor.ktensorflow.tensor.slice
import dev.kursor.ktensorflow.tensor.sum
import dev.kursor.ktensorflow.tensor.toArray
import dev.kursor.ktensorflow.tensor.toFlatIndex
import dev.kursor.ktensorflow.tensor.toFloatTensor
import dev.kursor.ktensorflow.tensor.toIntTensor
import dev.kursor.ktensorflow.tensor.transpose
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.expect

class TransformationsTest {

    @Test
    fun forEachTest() {
        val data = Array(2) { i -> FloatArray(3) { j -> (i * 3 + j).toFloat() } }
        val tensor = Tensor<Float>(data)
        val result = FloatArray(6)
        var index = 0;
        tensor.forEach {
            result[index++] = it
        }
        assertContentEquals(
            expected =  floatArrayOf(0f, 1f, 2f, 3f, 4f, 5f),
            actual = result
        )
    }

    @Test
    fun forEachIndexedTest() {
        val data = Array(2) { i -> FloatArray(3) { j -> (i * 3 + j).toFloat() } }
        val tensor = Tensor<Float>(data)
        val result = Array(2) { FloatArray(3) }
        tensor.forEachIndexed { index, it ->
            result[index[0]][index[1]] = it
        }
        assertContentDeepEquals(
            expected =  arrayOf(floatArrayOf(0f, 1f, 2f), floatArrayOf(3f, 4f, 5f)),
            actual = result
        )
    }

    @Test
    fun mapTest() {
        val data = Array(2) { i -> FloatArray(3) { j -> (i * 3 + j).toFloat() } }
        val tensor = Tensor<Float>(data)
        val result = tensor.map { it * 2 }.flatten().toArray<FloatArray>()
        assertContentEquals(
            expected =  floatArrayOf(0f, 2f, 4f, 6f, 8f, 10f),
            actual = result
        )
    }

    @Test
    fun mapIndexedTest() {
        val data = Array(2) { i -> FloatArray(3) { j -> (i * 3 + j).toFloat() } }
        val tensor = Tensor<Float>(data)
        val result = tensor.mapIndexed { index, it ->
            it * 2 + index.toFlatIndex(tensor.shape)
        }.flatten().toArray<FloatArray>()
        assertContentEquals(
            expected =  floatArrayOf(0f, 3f, 6f, 9f, 12f, 15f),
            actual = result
        )
    }

    @Test
    fun mapInPlaceTest() {
        val data = Array(2) { i -> FloatArray(3) { j -> (i * 3 + j).toFloat() } }
        val tensor = Tensor<Float>(data)
        tensor.mapInPlace { it * 2 }
        assertContentEquals(
            expected =  floatArrayOf(0f, 2f, 4f, 6f, 8f, 10f),
            actual = tensor.flatten().toArray()
        )
    }

    @Test
    fun mapInPlaceIndexedTest() {
        val data = Array(2) { i -> FloatArray(3) { j -> (i * 3 + j).toFloat() } }
        val tensor = Tensor<Float>(data)
        tensor.mapInPlaceIndexed { index, it ->
            it * 2 + index.toFlatIndex(tensor.shape)
        }
        assertContentEquals(
            expected =  floatArrayOf(0f, 3f, 6f, 9f, 12f, 15f),
            actual = tensor.flatten().toArray()
        )
    }

    @Test
    fun reshapeTest() {
        val data = Array(2) { i -> FloatArray(3) { j -> (i * 3 + j).toFloat() } }
        val tensor = Tensor<Float>(data)
        val result = tensor.reshape(TensorShape(3, 2)).toArray<Array<FloatArray>>()
        assertContentDeepEquals(
            expected =  arrayOf(floatArrayOf(0f, 1f), floatArrayOf(2f, 3f), floatArrayOf(4f, 5f)),
            actual = result
        )
    }

    @Test
    fun flattenTest() {
        val data = Array(2) { i -> FloatArray(3) { j -> (i * 3 + j).toFloat() } }
        val tensor = Tensor<Float>(data)
        val result = tensor.flatten().toArray<FloatArray>()
        assertContentEquals(
            expected =  floatArrayOf(0f, 1f, 2f, 3f, 4f, 5f),
            actual = result
        )
    }

    @Test
    fun transposeTest() {
        val data = Array(2) { i -> FloatArray(3) { j -> (i * 3 + j).toFloat() } }
        val tensor = Tensor<Float>(data)
        val result = tensor.transpose().toArray<Array<FloatArray>>()
        assertContentDeepEquals(
            expected =  arrayOf(floatArrayOf(0f, 3f), floatArrayOf(1f, 4f), floatArrayOf(2f, 5f)),
            actual = result
        )
    }

    @Test
    fun sliceTest() {
        val data = Array(2) { i -> FloatArray(3) { j -> (i * 3 + j).toFloat() } }
        val tensor = Tensor<Float>(data)
        val result = tensor.slice(arrayOf(0..1, 1..2)).toArray<Array<FloatArray>>()

        assertContentDeepEquals(
            expected =  arrayOf(floatArrayOf(1f, 2f), floatArrayOf(4f, 5f)),
            actual = result
        )
    }

    @Test
    fun sumTest() {
        val data = Array(2) { i -> FloatArray(3) { j -> (i * 3 + j).toFloat() } }
        val tensor = Tensor<Float>(data)
        val result = tensor.sum()
        assertEquals(15f, result)
    }

    @Test
    fun avgTest() {
        val data = Array(2) { i -> FloatArray(3) { j -> (i * 3 + j).toFloat() } }
        val tensor = Tensor<Float>(data)
        val result = tensor.avg()
        assertEquals(2.5f, result)
    }

    @Test
    fun toFloatTensorTest() {
        val data = Array(2) { i -> IntArray(3) { j -> i * 3 + j } }
        val tensor = Tensor<Int>(data)
        val result = tensor.toFloatTensor()
        assertContentDeepEquals(
            expected =  arrayOf(floatArrayOf(0f, 1f, 2f), floatArrayOf(3f, 4f, 5f)),
            actual = result.toArray()
        )
    }

    @OptIn(ExperimentalUnsignedTypes::class)
    @Test
    fun toIntTensorTest() {
        val data = Array(2) { i -> UByteArray(3) { j -> (i * 3 + j).toUByte() } }
        val tensor = Tensor<UByte>(data)
        val result = tensor.toIntTensor()
        assertContentDeepEquals(
            expected =  arrayOf(intArrayOf(0, 1, 2), intArrayOf(3, 4, 5)),
            actual = result.toArray()
        )
    }
}