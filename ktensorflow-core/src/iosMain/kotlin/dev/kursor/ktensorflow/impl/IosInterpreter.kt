package dev.kursor.ktensorflow.impl

import cocoapods.TensorFlowLiteObjC.TFLInterpreter
import cocoapods.TensorFlowLiteObjC.TFLInterpreterErrorCode
import cocoapods.TensorFlowLiteObjC.TFLTensor
import dev.kursor.ktensorflow.Interpreter
import dev.kursor.ktensorflow.InterpreterOptions
import dev.kursor.ktensorflow.ModelDesc
import dev.kursor.ktensorflow.ModelMeta
import dev.kursor.ktensorflow.ModelTensorData
import dev.kursor.ktensorflow.TensorFlowException
import dev.kursor.ktensorflow.toKTensorFlow
import platform.Foundation.NSRecursiveLock

// private val on options because it is required to keep references so that they are not
// garbage collected since kotlin gc does not know if objects are passed to obj-c
internal class IosInterpreter(
    modelDesc: ModelDesc,
    private val options: InterpreterOptions
) : Interpreter {

    // Замок на экземпляр: интерпретатор TensorFlow Lite не допускает одновременных вызовов,
    // а close() из другого потока во время инференса освобождал бы объект прямо под ним.
    // Рекурсивный, потому что публичные методы вызывают друг друга (getModelMeta -> inputTensorCount).
    private val lock = NSRecursiveLock()

    private inline fun <T> locked(block: () -> T): T {
        lock.lock()
        try {
            return block()
        } finally {
            lock.unlock()
        }
    }

    // null после close(): ссылка на TFLInterpreter отпускается, и нативный объект вместе с
    // моделью и тензорной ареной освобождается при ближайшей сборке мусора Kotlin/Native.
    // Раньше close() ничего не делал, а run после него продолжал работать, в отличие от Android.
    private var interpreterOrNull: TFLInterpreter? = checkError(
        // TensorFlowLiteObjC не говорит, почему интерпретатор не создан: тот же код ошибки
        // бывает и при операции, которой TensorFlow Lite не знает. С делегатами вероятнее всего
        // отказал один из них
        explain = { error ->
            val failedToCreate = error.code == TFLInterpreterErrorCode.TFLInterpreterErrorCodeFailedToCreateInterpreter.value.toLong()
            if (failedToCreate && options.tflDelegates.isNotEmpty()) DELEGATE_OR_OPERATION_FAILED_MESSAGE else null
        }
    ) { errPtr ->
        when (modelDesc) {
            is ModelDesc.PathInBundle -> {
                TFLInterpreter(
                    modelPath = modelDesc.pathInBundle,
                    options = options.tflOptions,
                    error = errPtr,
                    delegates = options.tflDelegates
                )
            }
        }
    }

    private val tflInterpreter: TFLInterpreter
        get() = interpreterOrNull ?: throw TensorFlowException("Interpreter has already been closed")

    // Метаданные неизменны до resizeInput, а именованный run запрашивает их на каждом вызове
    private var cachedMeta: ModelMeta? = null

    init {
        checkError { errPtr ->
            tflInterpreter.allocateTensorsWithError(errPtr)
        }
    }

    override val inputTensorCount: Int
        get() = locked { tflInterpreter.inputTensorCount.toInt() }

    override val outputTensorCount: Int
        get() = locked { tflInterpreter.outputTensorCount.toInt() }

    override fun getModelMeta(): ModelMeta = locked {
        cachedMeta ?: readModelMeta().also { cachedMeta = it }
    }

    private fun readModelMeta(): ModelMeta {
        val rawInputs = (0 until inputTensorCount).map { i ->
            i to getInputTensor(i)
        }
        val rawOutputs = (0 until outputTensorCount).map { i ->
            i to getOutputTensor(i)
        }

        @Suppress("UNCHECKED_CAST")
        val signatureKeys = tflInterpreter.signatureKeys() as? List<String>
        val defaultSignature = signatureKeys?.firstOrNull()

        if (defaultSignature != null) {
            val runner = checkError { err ->
                tflInterpreter.signatureRunnerWithKey(defaultSignature, err)
            }

            @Suppress("UNCHECKED_CAST")
            val sigInputs = runner.inputs() as? List<String> ?: emptyList()
            val inputs = sigInputs.map { sigName ->
                val sigTensor = checkError { err ->
                    runner.inputTensorWithName(sigName, err)
                }
                val internalName = sigTensor.name()
                val index = rawInputs.first { it.second.name() == internalName }.first

                ModelTensorData(
                    index = index,
                    name = sigName,
                    internalName = sigTensor.name(),
                    dataType = sigTensor.dataType().toKTensorFlow(),
                    shape = sigTensor.shape().map { (it as Number).toInt() }
                )
            }

            @Suppress("UNCHECKED_CAST")
            val sigOutputs = runner.outputs() as? List<String> ?: emptyList()
            val outputs = sigOutputs.map { sigName ->
                val sigTensor = checkError { err ->
                    runner.outputTensorWithName(sigName, err)
                }
                val internalName = sigTensor.name()
                val index = rawOutputs.first { it.second.name() == internalName }.first

                ModelTensorData(
                    index = index,
                    name = sigName,
                    internalName = sigTensor.name(),
                    dataType = sigTensor.dataType().toKTensorFlow(),
                    shape = sigTensor.shape().map { (it as Number).toInt() }
                )
            }

            return ModelMeta(inputs, outputs)
        }

        return ModelMeta(
            inputData = rawInputs.map { (index, tensor) ->
                ModelTensorData(
                    index = index,
                    name = tensor.name(),
                    internalName = tensor.name(),
                    dataType = tensor.dataType().toKTensorFlow(),
                    shape = tensor.shape().map { (it as Number).toInt() }
                )
            },
            outputData = rawOutputs.map { (index, tensor) ->
                ModelTensorData(
                    index = index,
                    name = tensor.name(),
                    internalName = tensor.name(),
                    dataType = tensor.dataType().toKTensorFlow(),
                    shape = tensor.shape().map { (it as Number).toInt() }
                )
            }
        )
    }

    private fun getInputTensor(index: Int): TFLTensor {
        return checkError { errPtr ->
            tflInterpreter.inputTensorAtIndex(index.toULong(), errPtr)
        }
    }

    private fun getOutputTensor(index: Int): TFLTensor {
        return checkError { errPtr ->
            tflInterpreter.outputTensorAtIndex(index.toULong(), errPtr)
        }
    }

    override fun resizeInput(index: Int, dims: IntArray): Unit = locked {
        cachedMeta = null
        resizeInputSafely(
            index = index,
            dims = dims,
            // shapeWithError сам отвергает скаляр и неположительные размерности: такую форму
            // нельзя вернуть через resizeInputTensorAtIndex, поэтому откатывать некуда
            currentShape = {
                getInputTensor(index).shapeWithError(null)
                    ?.map { (it as Number).toInt() }
                    ?.toIntArray()
            },
            // Здесь TensorFlowLiteObjC молча принимал форму, с которой выход получал нулевую
            // размерность, и отказ всплывал только на следующем run или getModelMeta
            outputShapesValid = {
                (0 until outputTensorCount).all { i -> getOutputTensor(i).shapeWithError(null) != null }
            },
            resizeAndAllocate = { shape ->
                checkError { errPtr ->
                    tflInterpreter.resizeInputTensorAtIndex(
                        index.toULong(),
                        shape.toList(),
                        errPtr
                    )
                }
                checkError<Boolean> { errPtr ->
                    tflInterpreter.allocateTensorsWithError(errPtr)
                }
            }
        )
    }

    override fun run(
        inputs: List<ByteArray>,
        outputs: Map<Int, ByteArray>
    ) = locked {
        // Ровно столько входов, сколько у модели, как требует Android: при меньшем числе
        // недостающие входы молча оставались от предыдущего вызова
        if (inputs.size != tflInterpreter.inputTensorCount().toInt()) {
            throw TensorFlowException(
                "Model has ${tflInterpreter.inputTensorCount()} inputs, got ${inputs.size}"
            )
        }

        inputs.forEachIndexed { index, input ->
            val inputTensor = getInputTensor(index)
            // copyData сам копирует байты в тензор, поэтому NSData лишь смотрит в массив
            input.withNSDataView { data ->
                checkError<Boolean> { errPtr ->
                    inputTensor.copyData(data, errPtr)
                }
            }
        }
        checkError { errPtr ->
            tflInterpreter.invokeWithError(errPtr)
        }

        for (entry in outputs) {
            val (i, byteArray) = entry
            val outputTensor = getOutputTensor(i)

            // dataWithError уже отдаёт копию данных тензора: из неё сразу в буфер вызывающего,
            // без промежуточного ByteArray, который раньше добавлял ещё две копии
            val data = checkError { errPtr ->
                outputTensor.dataWithError(errPtr)
            }

            // Короткий буфер раньше молча обрезал результат; Android в этом случае падает.
            // Буфер длиннее тензора допустим на обеих платформах: хвост остаётся нетронутым
            if (byteArray.size.toULong() < data.length) {
                throw TensorFlowException(
                    "Output $i holds ${data.length} bytes, the buffer has only ${byteArray.size}"
                )
            }
            data.copyInto(byteArray)
        }
    }

    override fun close() = locked {
        // Ждёт завершения идущего инференса. Повторное закрытие ничего не делает.
        interpreterOrNull = null
        cachedMeta = null
    }
}

internal fun TFLTensor.shape(): List<Int> {
    return checkError { errPtr ->
        this.shapeWithError(errPtr)
    }.map { (it as Number).toInt() }
}

/** Отказ создания на iOS: здесь нельзя отличить отказ делегата от неизвестной операции модели. */
private const val DELEGATE_OR_OPERATION_FAILED_MESSAGE =
    "Failed to create the interpreter: a delegate could not be applied to the model, " +
        "or the model uses an operation TensorFlow Lite does not support. " +
        "Create the interpreter without delegates to run the model on the CPU"
