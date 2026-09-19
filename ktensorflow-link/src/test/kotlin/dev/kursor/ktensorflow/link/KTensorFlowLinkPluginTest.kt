package dev.kursor.ktensorflow.link

import org.gradle.api.Project
import org.gradle.api.internal.project.ProjectInternal
import org.gradle.api.plugins.ExtensionAware
import org.gradle.testfixtures.ProjectBuilder
import org.jetbrains.kotlin.gradle.dsl.KotlinMultiplatformExtension
import org.jetbrains.kotlin.gradle.plugin.cocoapods.CocoapodsExtension
import org.jetbrains.kotlin.gradle.plugin.mpp.Framework
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Плагин собирается с KMP и cocoapods как compileOnly, поэтому несовместимость с версией
 * Kotlin у пользователя проявилась бы только в рантайме его сборки, а не при сборке здесь.
 * Эти тесты применяют плагин к настоящему проекту и проверяют, что он действительно
 * применяется и настраивает поды так, как рассчитывает iOS-линковка.
 */
class KTensorFlowLinkPluginTest {

    private val expectedTfLiteVersion = "2.17.0"

    private fun kmpProjectWithPlugin(): Project {
        val project = ProjectBuilder.builder().build()
        project.plugins.apply("org.jetbrains.kotlin.multiplatform")
        project.plugins.apply("org.jetbrains.kotlin.native.cocoapods")
        project.plugins.apply(KTensorFlowLinkPlugin::class.java)
        return project
    }

    private fun Project.cocoapods(): CocoapodsExtension {
        val kmp = extensions.getByType(KotlinMultiplatformExtension::class.java)
        return (kmp as ExtensionAware).extensions.getByType(CocoapodsExtension::class.java)
    }

    @Test
    fun pluginAppliesToAKotlinMultiplatformProject() {
        val project = kmpProjectWithPlugin()

        assertTrue(project.plugins.hasPlugin(KTensorFlowLinkPlugin::class.java))
    }

    @Test
    fun registersEveryTensorFlowLitePodNeededForLinking() {
        val project = kmpProjectWithPlugin()

        val podNames = project.cocoapods().pods.map { it.name }

        assertTrue(
            "TensorFlowLiteObjC pod is missing, iOS linking would fail: $podNames",
            podNames.contains("TensorFlowLiteObjC")
        )
        assertTrue(
            "Metal (GPU delegate) pod is missing: $podNames",
            podNames.contains("TensorFlowLiteObjC/Metal")
        )
        assertTrue(
            "CoreML (NPU delegate) pod is missing: $podNames",
            podNames.contains("TensorFlowLiteObjC/CoreML")
        )
    }

    @Test
    fun everyPodUsesTheSameModuleNameVersionAndIsLinkOnly() {
        val project = kmpProjectWithPlugin()

        project.cocoapods().pods.forEach { pod ->
            assertEquals("unexpected moduleName for ${pod.name}", "TFLTensorFlowLite", pod.moduleName)
            assertEquals("unexpected version for ${pod.name}", expectedTfLiteVersion, pod.version)
            assertTrue("${pod.name} must be linkOnly", pod.linkOnly)
        }
    }

    @Test
    fun linkerOptionsAreAddedToDynamicFrameworksOnly() {
        val project = kmpProjectWithPlugin()
        val kmp = project.extensions.getByType(KotlinMultiplatformExtension::class.java)

        // Фреймворки объявляются так же, как это делает пользователь: isStatic задаётся
        // внутри конфигурационного блока, уже после того как плагин подписался на binaries
        val simulator = kmp.iosSimulatorArm64()
        simulator.binaries.framework("dynamicFw") { isStatic = false }
        val device = kmp.iosArm64()
        device.binaries.framework("staticFw") { isStatic = true }

        // плагин принимает решение после конфигурации, поэтому проект нужно вычислить
        (project as ProjectInternal).evaluate()

        val frameworks = simulator.binaries.withType(Framework::class.java) +
            device.binaries.withType(Framework::class.java)

        val dynamic = frameworks.filter { !it.isStatic }
        val static = frameworks.filter { it.isStatic }

        assertTrue("test setup: no dynamic frameworks created", dynamic.isNotEmpty())
        assertTrue("test setup: no static frameworks created", static.isNotEmpty())

        dynamic.forEach {
            assertTrue(
                "dynamic framework ${it.name} must allow TensorFlow Lite symbols to stay " +
                    "undefined, got ${it.freeCompilerArgs}",
                it.freeCompilerArgs.contains("-linker-options")
            )
        }
        static.forEach {
            assertTrue(
                "static framework ${it.name} links TensorFlow Lite directly and needs no -U " +
                    "flags, got ${it.freeCompilerArgs}",
                it.freeCompilerArgs.none { arg -> arg == "-linker-options" }
            )
        }
    }

    @Test
    fun linkerArgumentsAllowTheTensorFlowLiteSymbolsToStayUndefined() {
        val arguments = linkerArgs.split(" ")

        // Каждый символ должен быть передан линковщику как -U <symbol>
        assertTrue("linker arguments must not be empty", arguments.isNotEmpty())
        assertEquals("every symbol must be preceded by -U", 0, arguments.size % 2)
        arguments.filterIndexed { index, _ -> index % 2 == 0 }.forEach { flag ->
            assertEquals("unexpected linker flag: $flag", "-U", flag)
        }

        // Символы, без которых не работает ни одна из заявленных возможностей
        listOf(
            "_TfLiteInterpreterCreate",
            "_TfLiteInterpreterInvoke",
            "_TfLiteSignatureRunnerInvoke",
            "_TFLGpuDelegateCreate",
            "_TfLiteCoreMlDelegateCreate",
            "_TfLiteXNNPackDelegateCreate"
        ).forEach { symbol ->
            assertTrue("symbol $symbol is not allowed to stay undefined", linkerArgs.contains("-U $symbol"))
        }
    }
}
