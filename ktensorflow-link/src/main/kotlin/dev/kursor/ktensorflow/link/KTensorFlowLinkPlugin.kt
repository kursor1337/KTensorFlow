package dev.kursor.ktensorflow.link

import org.gradle.api.Plugin
import org.gradle.api.Project
import org.gradle.kotlin.dsl.configure
import org.jetbrains.kotlin.gradle.dsl.KotlinMultiplatformExtension
import org.jetbrains.kotlin.gradle.plugin.cocoapods.CocoapodsExtension

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
        }
    }
}