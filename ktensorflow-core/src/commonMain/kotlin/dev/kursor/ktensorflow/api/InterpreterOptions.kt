package dev.kursor.ktensorflow.api

/**
 * Options for [Interpreter].
 * @param numThreads Number of threads to use for inference.
 * @param useXNNPACK Whether to use XNNPACK for optimized inference.
 * @param hardware [Hardware] to use for inference.
 */
data class InterpreterOptions(
    val numThreads: Int,
    val useXNNPACK: Boolean,
    val hardwarePriorities: List<Hardware> = Hardware.DefaultPriorities
)

/**
 * Hardware options for [Interpreter].
 */
enum class Hardware {
    /**
     * Use CPU for inference.
     */
    CPU,

    /**
     * Use GPU for inference.
     */
    GPU;

    /**
     * Use NPU for inference.
     * On Android is only available on devices with Qualcomm SoC.
     * On iOS is only available on devices with iOS ver > 12
     * and supports only Float32 data type.
     */
    // NPU;

    companion object {
        /**
         * Default priorities for hardware.
         * The order of the list is the order in which the hardware will be used.
         * If the hardware is not available, the next one will be used instead.
         */
        val DefaultPriorities = listOf(GPU, CPU)
    }
}
