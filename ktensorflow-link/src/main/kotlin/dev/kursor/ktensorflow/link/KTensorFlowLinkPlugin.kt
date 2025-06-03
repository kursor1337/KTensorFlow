package dev.kursor.ktensorflow.link

import org.gradle.api.Plugin
import org.gradle.api.Project
import org.gradle.kotlin.dsl.configure
import org.gradle.kotlin.dsl.withType
import org.jetbrains.kotlin.gradle.dsl.KotlinMultiplatformExtension
import org.jetbrains.kotlin.gradle.plugin.cocoapods.CocoapodsExtension
import org.jetbrains.kotlin.gradle.plugin.mpp.Framework
import org.jetbrains.kotlin.gradle.plugin.mpp.KotlinNativeTarget

class KTensorFlowLinkPlugin : Plugin<Project> {
    override fun apply(target: Project) = with(target) {
        extensions.configure<KotlinMultiplatformExtension> {
            extensions.configure<CocoapodsExtension> {
                val tflVersion = "2.17.0"

                pod("TensorFlowLiteObjC") {
                    moduleName = "TFLTensorFlowLite"
                    version = tflVersion
                    linkOnly = true
                }
                pod("TensorFlowLiteObjC/Metal") {
                    moduleName = "TFLTensorFlowLite"
                    version = tflVersion
                    linkOnly = true
                }
            }

            targets.withType(KotlinNativeTarget::class.java).configureEach {
                binaries.withType<Framework>().all {
                    if (!isStatic) {
                        freeCompilerArgs += listOf("-linker-options", linkerArgs)
                    }
                }
            }
        }
    }
}

val linkerArgs = listOf(
    "_TFLGpuDelegateCreate",
    "_TFLGpuDelegateDelete",
    "_TfLiteInterpreterAllocateTensors",
    "_TfLiteInterpreterCreate",
    "_TfLiteInterpreterDelete",
    "_TfLiteInterpreterGetInputTensor",
    "_TfLiteInterpreterGetInputTensorCount",
    "_TfLiteInterpreterGetOutputTensor",
    "_TfLiteInterpreterGetOutputTensorCount",
    "_TfLiteInterpreterGetSignatureCount",
    "_TfLiteInterpreterGetSignatureKey",
    "_TfLiteInterpreterGetSignatureRunner",
    "_TfLiteInterpreterInvoke",
    "_TfLiteInterpreterOptionsAddDelegate",
    "_TfLiteInterpreterOptionsCreate",
    "_TfLiteInterpreterOptionsDelete",
    "_TfLiteInterpreterOptionsSetErrorReporter",
    "_TfLiteInterpreterOptionsSetNumThreads",
    "_TfLiteInterpreterResizeInputTensor",
    "_TfLiteModelCreateFromFile",
    "_TfLiteModelDelete",
    "_TfLiteSignatureRunnerAllocateTensors",
    "_TfLiteSignatureRunnerDelete",
    "_TfLiteSignatureRunnerGetInputCount",
    "_TfLiteSignatureRunnerGetInputName",
    "_TfLiteSignatureRunnerGetInputTensor",
    "_TfLiteSignatureRunnerGetOutputCount",
    "_TfLiteSignatureRunnerGetOutputName",
    "_TfLiteSignatureRunnerGetOutputTensor",
    "_TfLiteSignatureRunnerInvoke",
    "_TfLiteSignatureRunnerResizeInputTensor",
    "_TfLiteTensorByteSize",
    "_TfLiteTensorCopyFromBuffer",
    "_TfLiteTensorData",
    "_TfLiteTensorDim",
    "_TfLiteTensorName",
    "_TfLiteTensorNumDims",
    "_TfLiteTensorQuantizationParams",
    "_TfLiteTensorType",
    "_TfLiteVersion",
    "_TfLiteXNNPackDelegateCreate",
    "_TfLiteXNNPackDelegateDelete",
    "_TfLiteXNNPackDelegateOptionsDefault",
    "___asan_after_dynamic_init",
    "___asan_before_dynamic_init",
    "___asan_handle_no_return",
    "___asan_init",
    "___asan_memcpy",
    "___asan_memset",
    "___asan_option_detect_stack_use_after_return",
    "___asan_register_image_globals",
    "___asan_report_load1",
    "___asan_report_load4",
    "___asan_report_load8",
    "___asan_report_store1",
    "___asan_report_store4",
    "___asan_report_store8",
    "___asan_shadow_memory_dynamic_address",
    "___asan_stack_malloc_0",
    "___asan_stack_malloc_1",
    "___asan_stack_malloc_2",
    "___asan_unregister_image_globals",
    "___asan_version_mismatch_check_apple_clang_1700"
).joinToString(" ") { "-U $it" }
