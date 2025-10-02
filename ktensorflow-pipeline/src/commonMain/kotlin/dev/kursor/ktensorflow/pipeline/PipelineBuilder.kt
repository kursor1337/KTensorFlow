package dev.kursor.ktensorflow.pipeline

import dev.kursor.ktensorflow.Interpreter
import dev.kursor.ktensorflow.Tensor
import dev.kursor.ktensorflow.TensorDataType
import dev.kursor.ktensorflow.TensorShape

class PipelineBuilder {
    private val inputs = mutableListOf<PipelineInput<Any>>()
    private val outputs = mutableListOf<PipelineOutput<Any>>()
    private var interpreter: Interpreter? = null
    private var outputCount = 0

    /**
     * Adds an input preprocessing branch to the pipeline.
     * Each call to `input()` corresponds to one model input tensor, in order.
     *
     * @param stage A [Stage] chain that must result in a [Tensor].
     */
    fun input(index: Int = inputs.size, stage: Stage<Any, Tensor>): PipelineBuilder {
        @Suppress("UNCHECKED_CAST")
        inputs.add(PipelineInput(index, stage))
        return this
    }

    /**
     * Sets the TensorFlow Lite interpreter to be used for inference.
     *
     * @param interpreter The configured [Interpreter] instance.
     */
    fun inference(interpreter: Interpreter): PipelineBuilder {
        this.interpreter = interpreter
        return this
    }

    /**
     * Adds an output postprocessing branch to the pipeline.
     * Each call to `output()` corresponds to one model output tensor, in order.
     *
     * @param stage A [Stage] chain that takes a [Tensor] as input.
     */
    fun output(
        index: Int = outputs.size,
        dataType: TensorDataType,
        shape: TensorShape,
        stage: Stage<Tensor, Any>
    ): PipelineBuilder {
        @Suppress("UNCHECKED_CAST")
        outputs.add(PipelineOutput(index, dataType, shape, stage))
        return this
    }

    /**
     * Constructs the [Pipeline] after verifying that all required components have been provided.
     *
     * @return An immutable, executable [Pipeline] instance.
     * @throws IllegalStateException if the interpreter is not set, or if the number of
     * input/output branches does not match the interpreter's signature.
     */
    fun build(): Pipeline {
        val interpreter = checkNotNull(interpreter) {
            "Interpreter must be set using .inference()"
        }

        check(inputs.size == interpreter.inputTensorCount) {
            "Mismatch: Builder has ${inputs.size} inputs, but interpreter expects ${interpreter.inputTensorCount}."
        }

        check(outputCount == interpreter.outputTensorCount) {
            "Mismatch: Builder has $outputCount outputs, but interpreter expects ${interpreter.outputTensorCount}."
        }

        return Pipeline(
            inputs = inputs,
            outputs = outputs,
            interpreter = interpreter
        )
    }
}

/*
    PipelineBuilder
        .input(
            Transformation
                .resize()
                .grayscale()
                .normalize()
        )
        .input(
            Transformation
                .lowercase()
                .removePunctuation()
                .trim()
                .tokenize()
        )
        .output(
            Transformation
                .argmax()
                .toClass(classes: List<String>)
        )
        .output()
        .output()
 */