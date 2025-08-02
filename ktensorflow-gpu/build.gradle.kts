import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlin.multiplatform)
    alias(libs.plugins.android.library)
    alias(libs.plugins.convention.publishing)
    alias(libs.plugins.kotlin.cocoapods)
}

kotlin {
    compilerOptions {
        optIn.addAll("kotlinx.cinterop.ExperimentalForeignApi")
    }

    androidTarget {
        compilerOptions {
            jvmTarget.set(JvmTarget.JVM_11)
        }
    }

    iosX64()
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

        pod("TensorFlowLiteObjC") {
            moduleName = "TFLTensorFlowLite"
            version = "2.17.0"
        }
        pod("TensorFlowLiteObjC/Metal") {
            moduleName = "TFLTensorFlowLite"
            version = "2.17.0"
        }
    }

    sourceSets {
        commonMain.dependencies {
            implementation(projects.ktensorflowCore)
        }
    }
}

android {
    namespace = "dev.kursor.ktensorflow.gpu"
    val compileVersion = libs.versions.android.compileSdk.get().toInt()
    val minVersion = libs.versions.android.minSdk.get().toInt()
    compileSdk = compileVersion
    defaultConfig {
        minSdk = minVersion
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
}
