package dev.kursor.ktensorflow

/**
 * Metadata for a machine learning model, describing its inputs and outputs.
 *
 * If the model has signatures, the tensors are those of its first signature, listed in the
 * signature's order and named by it; tensors of other signatures are not described. Otherwise
 * every tensor is listed in index order under its own name.
 *
 * The order of [inputData] is therefore not necessarily the order [Interpreter.run] expects its
 * inputs in: `run` takes inputs by index. Use [ModelTensorData.index] to place each input, for
 * example `inputData.sortedBy { it.index }`.
 *
 * @property inputData Input tensors of the model.
 * @property outputData Output tensors of the model.
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
 *
 * @property index Index of the tensor, as [Interpreter.run] addresses inputs and outputs.
 * @property name Name of the tensor in the model signature, or its own name without signatures.
 * @property internalName Name of the tensor in the model graph.
 * @property dataType Data type of the tensor elements.
 * @property shape Current shape of the tensor; it changes after [Interpreter.resizeInput].
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
     * Total size of the tensor data in bytes, or 0 for a [DataType.String] tensor, whose size
     * depends on its contents.
     */
    val totalByteSize: Int get() = numElements * dataType.byteSize
}
