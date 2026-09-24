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
        // Внутренний API модулей библиотеки: снаружи он требует явного opt-in
        optIn.add("dev.kursor.ktensorflow.InternalKTensorFlowApi")
    }

    android {
        namespace = "dev.kursor.ktensorflow.vision"
        compileSdk = libs.versions.android.compileSdk.get().toInt()
        minSdk = libs.versions.android.minSdk.get().toInt()
        compilerOptions {
            jvmTarget.set(JvmTarget.JVM_11)
        }
    }

    iosArm64()
    iosSimulatorArm64()

    sourceSets {
        androidMain.dependencies {
            implementation(libs.androidx.core)
        }
        commonMain.dependencies {
            implementation(projects.ktensorflowCore)
            implementation(projects.ktensorflowPipeline)

            // api, а не implementation: публичные сигнатуры модуля раскрывают типы
            // этих модулей, поэтому потребителям они нужны транзитивно
            api(projects.ktensorflowTensor)
        }
    }
}
