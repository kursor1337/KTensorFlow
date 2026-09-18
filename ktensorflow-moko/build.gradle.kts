import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlin.multiplatform)
    alias(libs.plugins.android.library)
    alias(libs.plugins.convention.publishing)
    alias(libs.plugins.convention.binary.compatibility)
}

kotlin {
    android {
        namespace = "dev.kursor.ktensorflow.moko"
        compileSdk = libs.versions.android.compileSdk.get().toInt()
        minSdk = libs.versions.android.minSdk.get().toInt()
        compilerOptions {
            jvmTarget.set(JvmTarget.JVM_11)
        }
    }

    iosArm64()
    iosSimulatorArm64()

    sourceSets {
        commonMain.dependencies {
            implementation(projects.ktensorflowCore)

            // api, а не implementation: публичные сигнатуры модуля принимают типы
            // FileResource и AssetResource, поэтому потребителям они нужны транзитивно
            api(libs.moko.resources)
        }
        androidMain.dependencies {
            implementation(libs.androidx.startup.runtime)
        }
    }
}
