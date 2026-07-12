import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlin.multiplatform)
    alias(libs.plugins.android.library)
    alias(libs.plugins.convention.publishing)
    alias(libs.plugins.binary.compatibility.validator)
    alias(libs.plugins.compose.compiler)
    alias(libs.plugins.compose.multiplatform)
}

kotlin {
    compilerOptions {
        optIn.addAll("kotlinx.cinterop.ExperimentalForeignApi")
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
            implementation(projects.ktensorflowTensor)
            implementation(projects.ktensorflowPipeline)

            implementation(libs.compose.ui)
            implementation(libs.compose.runtime)

        }
    }
}
