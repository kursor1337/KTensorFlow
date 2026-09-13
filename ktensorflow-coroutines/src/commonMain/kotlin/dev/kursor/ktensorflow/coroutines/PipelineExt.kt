package dev.kursor.ktensorflow.coroutines

import dev.kursor.ktensorflow.ExperimentalKTensorFlowApi
import dev.kursor.ktensorflow.pipeline.Pipeline
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.buffer
import kotlinx.coroutines.flow.channelFlow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.withContext


/**
 * Runs the pipeline with the given input.
 * This function runs the pipeline asynchronously on the default dispatcher.
 * @param input The input to the pipeline.
 * @return The output of the pipeline.
 */
@ExperimentalKTensorFlowApi
suspend fun <I, O> Pipeline<I, O>.runSuspend(input: I): O =
    withContext(Dispatchers.Default) {
        run(input)
    }

/**
 * Runs the pipeline with the given input flow.
 * This function runs the pipeline for every item in the input flow.
 * The pipeline is run asynchronously on the specified [dispatcher].
 * The default dispatcher is [Dispatchers.Default].
 *
 * @param inputFlow The input flow to the pipeline.
 * @param dispatcher The dispatcher to run the pipeline on.
 * @return The output flow of the pipeline.
 */
@ExperimentalKTensorFlowApi
fun <I, O> Pipeline<I, O>.processFlow(
    inputFlow: Flow<I>,
    dispatcher: CoroutineDispatcher = Dispatchers.Default
): Flow<O> = inputFlow
    .map { item -> run(item) }
    .flowOn(dispatcher)


/**
 * Runs the pipeline with the given input flow, dropping items if the pipeline is already running.
 * This function runs the pipeline for every item in the input flow.
 * If the pipeline is already running, the item is closed (using [AutoCloseable.close]) and the next item is processed.
 * The pipeline is run asynchronously on the [Dispatchers.Default].
 *
 * @param inputFlow The input flow to the pipeline.
 * @return The output flow of the pipeline.
 */
@ExperimentalKTensorFlowApi
fun <I : AutoCloseable, O> Pipeline<I, O>.processFlowDropping(
    inputFlow: Flow<I>,
): Flow<O> = channelFlow {
    val mutex = Mutex()

    inputFlow.collect { item ->
        if (mutex.tryLock()) {
            launch(Dispatchers.Default) {
                try {
                    val result = run(item)
                    send(result)
                } finally {
                    mutex.unlock()
                }
            }
        } else {
            item.close()
        }
    }
}.buffer(Channel.RENDEZVOUS)