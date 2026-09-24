import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlin.multiplatform)
    alias(libs.plugins.kotlin.cocoapods)
    alias(libs.plugins.android.library)
    alias(libs.plugins.convention.publishing)
    alias(libs.plugins.convention.binary.compatibility)
}

kotlin {
    compilerOptions {
        optIn.addAll("kotlinx.cinterop.ExperimentalForeignApi")
    }

    android {
        namespace = "dev.kursor.ktensorflow.core"
        compileSdk = libs.versions.android.compileSdk.get().toInt()
        minSdk = libs.versions.android.minSdk.get().toInt()
        compilerOptions {
            jvmTarget.set(JvmTarget.JVM_11)
        }
    }

    iosArm64()
    iosSimulatorArm64()

    cocoapods {
        summary = "Some description for the Shared Module"
        homepage = "Link to the Shared Module homepage"
        version = "1.0"
        ios.deploymentTarget = "13.0"
        framework {
            baseName = "shared"
            isStatic = true
        }

        // Это единственный cinterop для TensorFlowLiteObjC во всей библиотеке: он покрывает и
        // сабспеки Metal и CoreML, а ktensorflow-gpu и ktensorflow-npu берут привязки отсюда.
        // Если каждый модуль генерирует свой cinterop для того же пода, все они оказываются в
        // одном пакете cocoapods.TensorFlowLiteObjC, и итоговый iOS-бинарь с несколькими модулями
        // не линкуется ("symbol multiply defined", KT-46358; с Kotlin 2.4.20 - гарантированно).
        pod("TensorFlowLiteObjC") {
            moduleName = "TFLTensorFlowLite"
            version = "2.17.0"
        }
        pod("TensorFlowLiteObjC/Metal") {
            moduleName = "TFLTensorFlowLite"
            version = "2.17.0"
        }
        pod("TensorFlowLiteObjC/CoreML") {
            moduleName = "TFLTensorFlowLite"
            version = "2.17.0"
        }
    }

    sourceSets {
        androidMain.dependencies {
            api(libs.tensorflow)
        }
    }
}
