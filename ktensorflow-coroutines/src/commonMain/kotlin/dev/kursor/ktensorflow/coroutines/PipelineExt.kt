package dev.kursor.ktensorflow.coroutines

import dev.kursor.ktensorflow.ExperimentalKTensorFlowApi
import dev.kursor.ktensorflow.pipeline.Pipeline
import dev.kursor.ktensorflow.pipeline.Tuple
import dev.kursor.ktensorflow.pipeline.tuple
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.buffer
import kotlinx.coroutines.flow.channelFlow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.withContext
import kotlin.jvm.JvmName

/**
 * Runs the pipeline with the given input.
 * This function runs the pipeline asynchronously on the default dispatcher.
 * @param input The input to the pipeline.
 * @return The output of the pipeline.
 */
@ExperimentalKTensorFlowApi
suspend fun <I, O> Pipeline<I, O>.runSuspend(input: I): O =
    withContext(InferenceDispatcher) {
        run(input)
    }

/**
 * Runs the pipeline with the given input flow.
 * This function runs the pipeline for every item in the input flow.
 * The pipeline is run asynchronously on the specified [dispatcher].
 * By default inference is serialized, because the interpreter behind a pipeline is not
 * thread-safe; pass a dispatcher explicitly only if the pipeline does not run inference.
 *
 * @param inputFlow The input flow to the pipeline.
 * @param dispatcher The dispatcher to run the pipeline on.
 * @return The output flow of the pipeline.
 */
@ExperimentalKTensorFlowApi
fun <I, O> Pipeline<I, O>.processFlow(
    inputFlow: Flow<I>,
    dispatcher: CoroutineDispatcher = InferenceDispatcher
): Flow<O> = inputFlow
    .map { item -> run(item) }
    .flowOn(dispatcher)

/**
 * Runs a pipeline built with the pipeline builder (which accepts a single-element [Tuple.One])
 * over the given input flow.
 *
 * Each item is wrapped into [Tuple.One] before being passed to the pipeline, so the flow can be
 * collected directly from a camera or any other source without manual wrapping.
 * The pipeline is run asynchronously on the specified [dispatcher]. By default inference is
 * serialized, because the interpreter behind a pipeline is not thread-safe; pass a dispatcher
 * explicitly only if the pipeline does not run inference.
 *
 * @param inputFlow The input flow to the pipeline.
 * @param dispatcher The dispatcher to run the pipeline on.
 * @return The output flow of the pipeline.
 */
@ExperimentalKTensorFlowApi
@JvmName("processTupleFlow")
fun <I, O> Pipeline<Tuple.One<I>, O>.processFlow(
    inputFlow: Flow<I>,
    dispatcher: CoroutineDispatcher = InferenceDispatcher
): Flow<O> = inputFlow
    .map { item -> run(tuple(item)) }
    .flowOn(dispatcher)

/**
 * Runs the pipeline with the given input flow, dropping items if the pipeline is already running.
 * This function runs the pipeline for every item in the input flow.
 * If the pipeline is already running, the item is closed (using [AutoCloseable.close]) and the next item is processed.
 * The pipeline is run asynchronously, and inference is serialized to keep the
 * non-thread-safe interpreter safe.
 *
 * @param inputFlow The input flow to the pipeline.
 * @return The output flow of the pipeline.
 */
@ExperimentalKTensorFlowApi
fun <I : AutoCloseable, O> Pipeline<I, O>.processFlowDropping(
    inputFlow: Flow<I>,
): Flow<O> = processDropping(inputFlow) { item -> item.close() }

/**
 * Runs a pipeline built with the pipeline builder (which accepts a single-element [Tuple.One])
 * over the given input flow, dropping items if the pipeline is already running.
 *
 * Each item is wrapped into [Tuple.One] before being passed to the pipeline, so the flow can be
 * collected directly from a camera or any other source without manual wrapping.
 * If the pipeline is already running, the item is closed (using [AutoCloseable.close]) and the
 * next item is processed. The pipeline is run asynchronously, and inference is serialized
 * to keep the non-thread-safe interpreter safe.
 *
 * @param inputFlow The input flow to the pipeline.
 * @return The output flow of the pipeline.
 */
@ExperimentalKTensorFlowApi
@JvmName("processTupleFlowDropping")
fun <I : AutoCloseable, O> Pipeline<Tuple.One<I>, O>.processFlowDropping(
    inputFlow: Flow<I>,
): Flow<O> = processDropping(inputFlow.map { item -> tuple(item) }) { input -> input.first.close() }

@ExperimentalKTensorFlowApi
private fun <I, O> Pipeline<I, O>.processDropping(
    inputFlow: Flow<I>,
    release: (I) -> Unit
): Flow<O> = channelFlow {
    val mutex = Mutex()

    inputFlow.collect { item ->
        if (mutex.tryLock()) {
            launch(InferenceDispatcher) {
                try {
                    val result = run(item)
                    send(result)
                } finally {
                    mutex.unlock()
                }
            }
        } else {
            release(item)
        }
    }
}.buffer(Channel.RENDEZVOUS)