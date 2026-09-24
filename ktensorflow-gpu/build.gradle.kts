import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlin.multiplatform)
    alias(libs.plugins.android.library)
    alias(libs.plugins.convention.publishing)
    alias(libs.plugins.convention.binary.compatibility)
}

kotlin {
    compilerOptions {
        optIn.addAll("kotlinx.cinterop.ExperimentalForeignApi")
    }

    android {
        namespace = "dev.kursor.ktensorflow.gpu"
        compileSdk = libs.versions.android.compileSdk.get().toInt()
        minSdk = libs.versions.android.minSdk.get().toInt()
        compilerOptions {
            jvmTarget.set(JvmTarget.JVM_11)
        }
    }

    iosArm64()
    iosSimulatorArm64()

    // Собственного cinterop для TensorFlowLiteObjC здесь нет намеренно: привязки, включая
    // Metal, генерирует ktensorflow-core, и они приходят вместе с ним. Второй cinterop
    // того же пода ломает линковку iOS-приложения, подключающего несколько модулей.

    sourceSets {
        commonMain.dependencies {
            implementation(libs.tensorflow.gpu)

            // api, а не implementation: GpuDelegateOptions раскрывает наружу
            // GpuDelegateFactory.Options, и без этого потребитель не может ни вызвать
            // builder-перегрузку, ни прочитать tflOptions
            api(libs.tensorflow.gpu.api)

            // api, а не implementation: публичные сигнатуры модуля раскрывают типы
            // этих модулей, поэтому потребителям они нужны транзитивно
            api(projects.ktensorflowCore)
        }
    }
}
