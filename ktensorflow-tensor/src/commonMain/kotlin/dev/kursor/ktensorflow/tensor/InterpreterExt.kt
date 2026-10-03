package dev.kursor.ktensorflow.tensor

import dev.kursor.ktensorflow.Interpreter
import dev.kursor.ktensorflow.ModelTensorData
import dev.kursor.ktensorflow.TensorFlowException
import kotlin.jvm.JvmName

/**
 * Runs model inference for multiple inputs and outputs.
 * Result of the inference will be written to the output [Tensor]s, which should be
 * allocated beforehand and passed to this method.
 *
 * Calls from several threads are safe: the interpreter serializes them.
 *
 * @param inputs List of input [Tensor]s.
 * @param outputs Map of output [Tensor]s, key is index of the output [Tensor]
 * @throws TensorFlowException if a tensor's data type differs from the model's, the
 *   buffers do not match the model or inference fails.
 */
fun Interpreter.run(
    inputs: List<PhysicalTensor<*>>,
    outputs: Map<Int, PhysicalTensor<*>>
) {
    // TensorFlow Lite получает только байты и тип не видит: Tensor<Int> на вход Float32-модели
    // того же размера молча читался как float, а выход квантованной модели, объявленный как
    // Float32, превращал байты в мусорные числа. Метаданные кешируются, проверка дешёвая
    val meta = getModelMeta()
    inputs.forEachIndexed { index, tensor -> meta.inputData.requireType(index, tensor, "input") }
    outputs.forEach { (index, tensor) -> meta.outputData.requireType(index, tensor, "output") }

    run(
        inputs = inputs.map { it.data },
        outputs = outputs.mapValues { it.value.data }
    )
}

private fun List<ModelTensorData>.requireType(index: Int, tensor: PhysicalTensor<*>, kind: String) {
    // Несуществующий индекс здесь не проверяется: о нём сообщит сам интерпретатор
    val expected = firstOrNull { it.index == index } ?: return
    if (TensorDataType.of(expected.dataType) != tensor.dataType) {
        throw TensorFlowException(
            "Model $kind $index ('${expected.name}') holds ${expected.dataType} data, " +
                "got a ${tensor.dataType} tensor"
        )
    }
}

/**
 * Runs model inference for multiple inputs and outputs.
 * Result of the inference will be written to the output [Tensor]s, which should be
 * allocated beforehand and passed to this method.
 *
 * Calls from several threads are safe: the interpreter serializes them.
 *
 * @param inputs List of input [Tensor]s.
 * @param outputs Map of output [Tensor]s, key is index of the output [Tensor]
 * @throws TensorFlowException if a tensor's data type differs from the model's, the
 *   buffers do not match the model or inference fails.
 */
@JvmName("runWithNames")
fun Interpreter.run(
    inputs: List<PhysicalTensor<*>>,
    outputs: Map<String, PhysicalTensor<*>>
) {
    val modelMeta = getModelMeta()
    // Неизвестное имя раньше молча превращалось в индекс 0: результат писался не в тот
    // буфер, а два неизвестных имени ещё и схлопывались в один ключ, теряя выход целиком
    val outputMap = outputs.mapKeys { (name, _) ->
        val output = modelMeta.outputsByName[name]
        requireNotNull(output) {
            "Model has no output named '$name'. Available outputs: " +
                modelMeta.outputsByName.keys.joinToString()
        }
        output.index
    }
    run(
        inputs = inputs,
        outputs = outputMap
    )
}

/**
 * Runs model inference for single input and output.
 * Result of the inference will be written to the output [Tensor], which should be
 * allocated beforehand and passed to this method.
 *
 * Calls from several threads are safe: the interpreter serializes them.
 *
 * @param input input [Tensor].
 * @param output output [Tensor]
 * @throws TensorFlowException if a tensor's data type differs from the model's, the
 *   buffers do not match the model or inference fails.
 */
fun Interpreter.run(
    input: PhysicalTensor<*>,
    output: PhysicalTensor<*>
) = run(listOf(input), mapOf(0 to output))