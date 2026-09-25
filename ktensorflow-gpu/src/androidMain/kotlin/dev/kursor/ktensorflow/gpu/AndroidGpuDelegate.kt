package dev.kursor.ktensorflow.gpu

import org.tensorflow.lite.Delegate
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegateFactory

internal class AndroidGpuDelegate(
    val options: GpuDelegateFactory.Options
) : GpuDelegate {

    // CompatibilityList держит нативный объект без финализатора: без close он утекал
    // с каждым созданным делегатом
    override val isAvailable: Boolean = CompatibilityList().use {
        it.isDelegateSupportedOnThisDevice
    }

    private var delegate: org.tensorflow.lite.gpu.GpuDelegate? = null
    private var closed = false

    // Нативный делегат создаётся лениво, при первом интерпретаторе: конструктор сразу
    // заводит GPU-контекст, и недоступный делегат не должен его трогать
    override val tflDelegate: Delegate
        @Synchronized get() {
            check(!closed) { "GPU delegate has already been closed" }
            return delegate
                ?: org.tensorflow.lite.gpu.GpuDelegate(options)
                    .also { delegate = it }
        }

    @Synchronized
    override fun close() {
        closed = true
        delegate?.close()
        delegate = null
    }
}
