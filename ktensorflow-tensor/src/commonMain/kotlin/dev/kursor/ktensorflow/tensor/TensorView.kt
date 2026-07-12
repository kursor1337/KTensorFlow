package dev.kursor.ktensorflow.tensor

interface TensorView<T : Any> : Tensor<T> {
    val delegate: Tensor<T>

    override fun toPhysical(): PhysicalTensor<T> {
        val result = Tensor(dataType, shape)
        for (i in 0 until shape.flatSize) {
            result.setFlat(i, this.getFlat(i))
        }
        return result
    }
}