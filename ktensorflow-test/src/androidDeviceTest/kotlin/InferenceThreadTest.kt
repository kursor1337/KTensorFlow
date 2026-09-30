import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import dev.kursor.ktensorflow.ExperimentalKTensorFlowApi
import dev.kursor.ktensorflow.InterpreterOptions
import dev.kursor.ktensorflow.coroutines.InferenceDispatcher
import dev.kursor.ktensorflow.coroutines.SuspendInterpreter
import dev.kursor.ktensorflow.coroutines.runSuspend
import dev.kursor.ktensorflow.pipeline.Pipeline
import dev.kursor.ktensorflow.pipeline.stage.Stage
import dev.kursor.ktensorflow.tensor.Tensor
import dev.kursor.ktensorflow.tensor.TensorShape
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import org.junit.Test
import org.junit.runner.RunWith
import java.util.concurrent.ConcurrentHashMap
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue
import kotlin.time.Duration.Companion.milliseconds
import kotlin.time.TimeSource

/**
 * GPU-делегат на OpenGL (Android без OpenCL) требует инференса на том потоке, где создан
 * интерпретатор. Раньше модуль корутин выполнял вызовы по одному, но на любом потоке пула.
 */
@OptIn(ExperimentalKTensorFlowApi::class)
@RunWith(AndroidJUnit4::class)
class InferenceThreadTest {

    private val context = InstrumentationRegistry.getInstrumentation().targetContext

    @Test
    fun everyInferenceThroughTheModuleRunsOnTheThreadThatCreatesInterpreters() = runBlocking {
        val threads = ConcurrentHashMap.newKeySet<Thread>()
        val pipeline = Pipeline(Stage<Int, Int> { threads += Thread.currentThread(); it })

        repeat(5) { pipeline.runSuspend(it) }
        withContext(Dispatchers.IO) { repeat(5) { pipeline.runSuspend(it) } }
        // Много параллельных вызовов, пока пул занят другой работой: на пуле потоков вызовы
        // разошлись бы по разным рабочим потокам даже при выполнении по одному
        coroutineScope {
            repeat(8) {
                launch(Dispatchers.Default) {
                    val end = TimeSource.Monotonic.markNow() + 200.milliseconds
                    while (end.hasNotPassedNow()) Unit
                }
            }
            repeat(50) { i ->
                launch(Dispatchers.Default) {
                    delay((i % 5).toLong())
                    pipeline.runSuspend(i)
                }
            }
        }

        assertEquals(1, threads.size, "inference must stay on one thread, was $threads")
    }

    @Test
    fun suspendInterpreterGivesAWorkingInterpreter() = runBlocking {
        val interpreter = SuspendInterpreter(loadModel(context, "mnist.tflite"), InterpreterOptions())
        val output = Tensor<Float>(shape = TensorShape(10)).toPhysical()

        interpreter.runSuspend(Tensor<Float>(shape = TensorShape(28, 28)).toPhysical(), output)

        assertTrue((0 until 10).all { output.getFlat(it).isFinite() }, "the model must have produced its outputs")
        interpreter.close()
    }

    @Test
    fun anInterpreterOnItsOwnInferenceDispatcherIsCreatedAndRunOnItsThread() = runBlocking {
        // Свой InferenceDispatcher - способ запускать модели параллельно, сохраняя требование
        // OpenGL-бэкенда GPU-делегата: создание и все запуски на одном потоке, отдельном от общего
        val threadOf = { dispatcher: InferenceDispatcher? ->
            val threads = ConcurrentHashMap.newKeySet<Thread>()
            val pipeline = Pipeline(Stage<Int, Int> { threads += Thread.currentThread(); it })
            runBlocking {
                repeat(5) { if (dispatcher == null) pipeline.runSuspend(it) else pipeline.runSuspend(it, dispatcher) }
            }
            threads.single()
        }
        val dispatcher = InferenceDispatcher()

        val interpreter = SuspendInterpreter(loadModel(context, "mnist.tflite"), InterpreterOptions(), dispatcher)
        val output = Tensor<Float>(shape = TensorShape(10)).toPhysical()
        interpreter.runSuspend(Tensor<Float>(shape = TensorShape(28, 28)).toPhysical(), output, dispatcher)
        assertTrue((0 until 10).all { output.getFlat(it).isFinite() })
        interpreter.close()

        val own = threadOf(dispatcher)
        assertTrue(own !== threadOf(null), "an own dispatcher must not share the default thread")

        dispatcher.close()
        own.join(5_000)
        assertFalse(own.isAlive, "closing the dispatcher must stop its thread")
    }
}
