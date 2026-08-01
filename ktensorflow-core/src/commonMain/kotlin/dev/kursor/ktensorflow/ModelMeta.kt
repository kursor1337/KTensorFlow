package dev.kursor.ktensorflow

/**
 * Metadata for a machine learning model, describing its inputs and outputs.
 */
data class ModelMeta(
    val inputData: List<ModelTensorData>,
    val outputData: List<ModelTensorData>
) {
    /**
     * Map of input tensors indexed by their name.
     */
    val inputsByName: Map<String, ModelTensorData> by lazy { inputData.associateBy { it.name } }

    /**
     * Map of output tensors indexed by their name.
     */
    val outputsByName: Map<String, ModelTensorData> by lazy { outputData.associateBy { it.name } }
}

/**
 * Information about a single input or output tensor of a model.
 */
data class ModelTensorData(
    val index: Int,
    val name: String,
    val internalName: String,
    val dataType: DataType,
    val shape: List<Int>
) {
    /**
     * Total number of elements in the tensor.
     */
    val numElements: Int get() = shape.fold(1) { acc, i -> acc * i }

    /**
     * Total size of the tensor data in bytes.
     */
    val totalByteSize: Int get() = numElements * dataType.byteSize
}
