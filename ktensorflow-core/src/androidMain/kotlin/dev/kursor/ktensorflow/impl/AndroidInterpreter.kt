package dev.kursor.ktensorflow.impl

import dev.kursor.ktensorflow.Interpreter
import dev.kursor.ktensorflow.InterpreterOptions
import dev.kursor.ktensorflow.ModelDesc
import dev.kursor.ktensorflow.ModelMeta
import dev.kursor.ktensorflow.ModelTensorData
import dev.kursor.ktensorflow.TensorFlowException
import dev.kursor.ktensorflow.toKTensorFlow
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.tensorflow.lite.Interpreter as TFLInterpreter

internal class AndroidInterpreter(
    modelDesc: ModelDesc,
    options: InterpreterOptions
) : Interpreter {

    private val tensorFlowInterpreter = tensorFlowCall("load the model") {
        when (modelDesc) {
            is ModelDesc.ByteBuffer -> TFLInterpreter(
                modelDesc.buffer.asDirect(),
                options.tflOptions,
            )

            is ModelDesc.File -> TFLInterpreter(
                modelDesc.file,
                options.tflOptions,
            )
        }
    }

    private val lock = Any()
    private var closed = false

    // Метаданные неизменны до resizeInput, а именованный run запрашивает их на каждом вызове
    private var cachedMeta: ModelMeta? = null

    private inline fun <T> locked(block: () -> T): T = synchronized(lock) {
        if (closed) throw TensorFlowException("Interpreter has already been closed")
        block()
    }

    override val inputTensorCount: Int
        get() = locked { tensorFlowInterpreter.inputTensorCount }

    override val outputTensorCount: Int
        get() = locked { tensorFlowInterpreter.outputTensorCount }

    override fun getModelMeta(): ModelMeta = locked {
        cachedMeta ?: readModelMeta().also { cachedMeta = it }
    }

    private fun readModelMeta(): ModelMeta = tensorFlowCall("read the model metadata") {
        val rawInputs = (0 until inputTensorCount).map { i ->
            i to tensorFlowInterpreter.getInputTensor(i)
        }
        val rawOutputs = (0 until outputTensorCount).map { i ->
            i to tensorFlowInterpreter.getOutputTensor(i)
        }

        val signatureKeys = tensorFlowInterpreter.signatureKeys
        val defaultSignature = signatureKeys.firstOrNull()

        if (defaultSignature != null) {
            val inputs = tensorFlowInterpreter
                .getSignatureInputs(defaultSignature)
                .map { sigName ->
                    val sigTensor = tensorFlowInterpreter
                        .getInputTensorFromSignature(
                            sigName,
                            defaultSignature
                        )

                    val index = rawInputs
                        .first { it.second.name() == sigTensor.name() }
                        .first

                    ModelTensorData(
                        index = index,
                        name = sigName,
                        internalName = sigTensor.name(),
                        dataType = sigTensor.dataType().toKTensorFlow(),
                        shape = sigTensor.shape().toList()
                    )
                }

            val outputs = tensorFlowInterpreter
                .getSignatureOutputs(defaultSignature)
                .map { sigName ->
                    val sigTensor = tensorFlowInterpreter
                        .getOutputTensorFromSignature(
                            sigName,
                            defaultSignature
                        )

                    val index = rawOutputs
                        .first { it.second.name() == sigTensor.name() }
                        .first

                    ModelTensorData(
                        index = index,
                        name = sigName,
                        internalName = sigTensor.name(),
                        dataType = sigTensor.dataType().toKTensorFlow(),
                        shape = sigTensor.shape().toList()
                    )
                }

            return@tensorFlowCall ModelMeta(inputs, outputs)
        }

        ModelMeta(
            inputData = rawInputs.map { (index, tensor) ->
                ModelTensorData(
                    index = index,
                    name = tensor.name(),
                    internalName = tensor.name(),
                    dataType = tensor.dataType().toKTensorFlow(),
                    shape = tensor.shape().toList()
                )
            },
            outputData = rawOutputs.map { (index, tensor) ->
                ModelTensorData(
                    index = index,
                    name = tensor.name(),
                    internalName = tensor.name(),
                    dataType = tensor.dataType().toKTensorFlow(),
                    shape = tensor.shape().toList()
                )
            }
        )
    }

    override fun resizeInput(index: Int, dims: IntArray) = locked {
        tensorFlowCall("resize input $index") {
            tensorFlowInterpreter.resizeInput(index, dims)
            tensorFlowInterpreter.allocateTensors()
        }
        cachedMeta = null
    }

    @OptIn(ExperimentalUnsignedTypes::class)
    override fun run(inputs: List<ByteArray>, outputs: Map<Int, ByteArray>) {
        val inputsArray = inputs
            .map {
                ByteBuffer.wrap(it)
                    .apply { order(ByteOrder.nativeOrder()) }
            }
            .toTypedArray()
        val outputsArray = outputs.mapValues {
            ByteBuffer.wrap(it.value)
                .apply { order(ByteOrder.nativeOrder()) }
        }

        locked {
            tensorFlowCall("run inference") {
                tensorFlowInterpreter.runForMultipleInputsOutputs(
                    inputsArray,
                    outputsArray
                )
            }
        }
    }

    override fun close() = synchronized(lock) {
        if (!closed) {
            closed = true
            cachedMeta = null
            tensorFlowCall("close the interpreter") { tensorFlowInterpreter.close() }
        }
    }
}

/**
 * TensorFlow Lite принимает модель только в direct- или отображённом буфере: обычный
 * ByteBuffer.wrap(bytes) отклонялся с "Failed to load the model". Такой буфер один раз
 * копируется в нативную память; позиция буфера вызывающего не меняется.
 */
private fun ByteBuffer.asDirect(): ByteBuffer =
    if (isDirect) {
        this
    } else {
        ByteBuffer.allocateDirect(remaining())
            .order(ByteOrder.nativeOrder())
            .put(duplicate())
            .apply { rewind() }
    }
