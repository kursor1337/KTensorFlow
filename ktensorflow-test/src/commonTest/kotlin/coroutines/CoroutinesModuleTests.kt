package coroutines

import dev.kursor.ktensorflow.ExperimentalKTensorFlowApi
import dev.kursor.ktensorflow.coroutines.mapAndClose
import dev.kursor.ktensorflow.coroutines.processFlow
import dev.kursor.ktensorflow.coroutines.processFlowDropping
import dev.kursor.ktensorflow.coroutines.runSuspend
import dev.kursor.ktensorflow.pipeline.Pipeline
import dev.kursor.ktensorflow.pipeline.stage.Stage
import kotlinx.coroutines.flow.asFlow
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.test.runTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

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

        items.forEach { item ->
            if (item.id in results) {
                assertFalse(item.closed, "a processed item must not be closed by processFlowDropping itself")
            } else {
                assertTrue(item.closed, "a dropped item must be closed to avoid leaking its resources")
            }
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
}
