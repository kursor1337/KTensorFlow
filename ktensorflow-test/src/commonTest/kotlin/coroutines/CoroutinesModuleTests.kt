package coroutines

import dev.kursor.ktensorflow.ExperimentalKTensorFlowApi
import dev.kursor.ktensorflow.coroutines.mapAndClose
import dev.kursor.ktensorflow.coroutines.processFlow
import dev.kursor.ktensorflow.coroutines.processFlowDropping
import dev.kursor.ktensorflow.coroutines.runSuspend
import dev.kursor.ktensorflow.pipeline.Pipeline
import dev.kursor.ktensorflow.pipeline.Tuple
import dev.kursor.ktensorflow.pipeline.stage.Stage
import dev.kursor.ktensorflow.pipeline.tuple
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.asFlow
import kotlinx.coroutines.flow.emptyFlow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.withTimeout
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue
import kotlin.time.Duration.Companion.milliseconds

/**
 * These tests exercise the real [Pipeline]/[Flow] machinery (kotlinx.coroutines is a real,
 * production dependency, not mocked) with lightweight pure-Kotlin stages. No platform/model
 * dependency is needed here since none of this module's logic touches Interpreter internals -
 * see [InterpreterTest] (android/iOS) for coroutine behavior against a real interpreter.
 */
@OptIn(ExperimentalKTensorFlowApi::class)
class CoroutinesModuleTests {

    private class TrackedItem(val id: Int) : AutoCloseable {
        var closed = false
            private set

        override fun close() {
            closed = true
        }
    }

    @Test
    fun pipelineRunSuspendProducesTheSameResultAsTheSynchronousRun() = runTest {
        val pipeline = Pipeline(Stage<Int, Int> { it + 1 })

        assertEquals(pipeline.run(41), pipeline.runSuspend(41))
    }

    @Test
    fun processFlowMapsEveryItemPreservingOrder() = runTest {
        val pipeline = Pipeline(Stage<Int, Int> { it * 2 })

        val result = pipeline.processFlow(flowOf(1, 2, 3, 4, 5)).toList()

        assertEquals(listOf(2, 4, 6, 8, 10), result)
    }

    @Test
    fun processFlowWorksWithSingleInputBuilderPipelines() = runTest {
        // Pipeline.input(...).inference(...).output(...).build() строит Pipeline<Tuple.One<Input>, ...>,
        // поэтому Flow<Input> должен приниматься без ручной обёртки в tuple() - так же, как в
        // processFlowDropping.
        val received = mutableListOf<Tuple.One<Int>>()
        val pipeline = Pipeline(Stage<Tuple.One<Int>, Int> { input ->
            received += input
            input.first * 2
        })

        val result = pipeline.processFlow(flowOf(1, 2, 3)).toList()

        assertEquals(listOf(2, 4, 6), result)
        assertEquals(listOf(tuple(1), tuple(2), tuple(3)), received)
    }

    @Test
    fun processFlowDroppingProcessesTheFirstItemAndDropsTheRestWhileBusy() = runTest {
        // Симулируем медленную (например, реальную ML) обработку блокирующей задержкой -
        // run() не suspend-функция, поэтому используем настоящий блокирующий вызов.
        val pipeline = Pipeline(Stage<TrackedItem, Int> { item ->
            runBlocking { kotlinx.coroutines.delay(200) }
            item.id
        })

        val items = (1..5).map { TrackedItem(it) }

        // items.asFlow() эмитит все элементы практически мгновенно, без suspend-задержек,
        // так что все они успевают дойти до processFlowDropping, пока обрабатывается первый.
        val results = pipeline.processFlowDropping(items.asFlow()).toList()

        assertEquals(listOf(1), results, "only the first item should win the mutex and be processed")

        // Поток владеет каждым кадром: отброшенные закрыты сразу, обработанный - после пайплайна
        items.forEach { item ->
            assertTrue(item.closed, "item ${item.id} must be closed by processFlowDropping")
        }
    }

    @Test
    fun processFlowDroppingWorksWithSingleInputBuilderPipelines() = runTest {
        // Pipeline.input(...).inference(...).output(...).build() строит Pipeline<Tuple.One<Input>, ...>.
        // Здесь тот же тип собран напрямую, чтобы не требовать интерпретатора и модели.
        val received = mutableListOf<Tuple.One<TrackedItem>>()
        val pipeline = Pipeline(Stage<Tuple.One<TrackedItem>, Int> { input ->
            received += input
            runBlocking { kotlinx.coroutines.delay(200) }
            input.first.id
        })

        val items = (1..5).map { TrackedItem(it) }

        // На вход подаётся Flow<TrackedItem> без ручной обёртки в tuple()
        val results = pipeline.processFlowDropping(items.asFlow()).toList()

        assertEquals(listOf(1), results, "only the first item should win the mutex and be processed")
        assertEquals(listOf(tuple(items[0])), received, "the pipeline must receive the item wrapped in Tuple.One")

        items.forEach { item ->
            assertTrue(item.closed, "item ${item.id} must be closed even though it was wrapped in a tuple")
        }
    }

    @Test
    fun processFlowDroppingClosesAnItemWhosePipelineFailed() = runTest {
        val item = TrackedItem(1)
        val failing = Pipeline(Stage<TrackedItem, Int> { error("model failed") })

        assertFailsWith<IllegalStateException> { failing.processFlowDropping(flowOf(item)).toList() }

        assertTrue(item.closed, "an item must be closed even when the pipeline throws on it")
    }

    @Test
    fun processFlowDroppingClosesTheItemInFlightWhenTheUpstreamFails() = runTest {
        val first = TrackedItem(1)
        val pipeline = Pipeline(Stage<TrackedItem, Int> { it.id })
        val broken = flow { emit(first); error("camera died") }

        assertFailsWith<IllegalStateException> { pipeline.processFlowDropping(broken).toList() }

        assertTrue(first.closed, "the item accepted before the upstream failed must be closed")
    }

    @Test
    fun processFlowDroppingFailsInsteadOfHangingWhenClosingAnItemThrows() = runBlocking {
        // Если close() бросает, mutex остаётся занятым - но зависания нет: исключение роняет
        // весь поток, и следующих кадров, которые упёрлись бы в mutex, уже не будет
        class BrokenItem : AutoCloseable {
            override fun close() = error("close failed")
        }
        val pipeline = Pipeline(Stage<BrokenItem, Int> { 1 })
        val frames = flow {
            repeat(5) {
                emit(BrokenItem())
                delay(20.milliseconds)
            }
        }

        val failure = withTimeout(5_000.milliseconds) {
            assertFailsWith<IllegalStateException> { pipeline.processFlowDropping(frames).toList() }
        }

        assertEquals("close failed", failure.message)
    }

    @Test
    fun processFlowDroppingOfAnEmptyFlowCompletesWithoutResults() = runTest {
        val pipeline = Pipeline(Stage<TrackedItem, Int> { it.id })

        assertEquals(emptyList(), pipeline.processFlowDropping(emptyFlow()).toList())
    }

    @Test
    fun itemAcceptedButCancelledBeforeInferenceStartsIsClosed() = runBlocking {
        withTimeout(15_000.milliseconds) {
            // Занимаем единственный слот InferenceDispatcher, чтобы инференс для item встал в очередь
            val gate = Channel<Unit>()
            val blocker = launch(Dispatchers.Default) {
                Pipeline(Stage<Unit, Unit> { runBlocking { gate.receive() } }).runSuspend(Unit)
            }
            delay(100.milliseconds)

            var processed = false
            val item = TrackedItem(1)
            val pipeline = Pipeline(Stage<TrackedItem, Int> { processed = true; it.id })
            val collector = launch(Dispatchers.Default) {
                pipeline.processFlowDropping(flowOf(item)).collect { }
            }
            delay(100.milliseconds)

            // join сразу нельзя: отменённая корутина завершится, только когда её возьмёт диспетчер
            collector.cancel()
            gate.send(Unit)
            blocker.join()
            collector.join()

            assertFalse(processed, "a cancelled collection must not run the pipeline")
            assertTrue(item.closed, "an item that was accepted but never processed must still be closed")
        }
    }

    @Test
    fun mapAndCloseClosesEachItemAfterASuccessfulTransform() = runTest {
        val item = TrackedItem(1)

        val result = flowOf(item).mapAndClose { it.id * 10 }.toList()

        assertEquals(listOf(10), result)
        assertTrue(item.closed)
    }

    @Test
    fun mapAndCloseClosesTheItemEvenWhenTheTransformThrows() = runTest {
        val item = TrackedItem(1)

        val flow = flowOf(item).mapAndClose<TrackedItem, Int> { error("boom") }

        assertFailsWith<IllegalStateException> { flow.toList() }
        assertTrue(item.closed)
    }

    @Test
    fun concurrentRunSuspendCallsNeverOverlap() = runTest {
        // Interpreter.run не потокобезопасен, поэтому два инференса не должны выполняться
        // одновременно, даже если пользователь запустил их параллельно. Пересечение ловится
        // мьютексом: если tryLock не удался, значит внутрь уже кто-то вошёл.
        val insideInference = Mutex()
        var overlapped = false

        val pipeline = Pipeline(Stage<Int, Int> { value ->
            if (!insideInference.tryLock()) {
                overlapped = true
            } else {
                runBlocking { kotlinx.coroutines.delay(50) }
                insideInference.unlock()
            }
            value * 2
        })

        val results = coroutineScope {
            (1..8).map { value -> async { pipeline.runSuspend(value) } }.awaitAll()
        }

        assertFalse(overlapped, "inference must never run concurrently: Interpreter.run is not thread-safe")
        assertEquals(listOf(2, 4, 6, 8, 10, 12, 14, 16), results)
    }
}
