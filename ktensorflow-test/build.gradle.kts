import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlin.multiplatform)
    alias(libs.plugins.kotlin.cocoapods)
    alias(libs.plugins.android.library)
}

kotlin {
    androidTarget {
        compilerOptions {
            jvmTarget.set(JvmTarget.JVM_11)
        }
    }

    iosX64()
    iosArm64()
    iosSimulatorArm64 {
        binaries {
            getTest("DEBUG").apply {
                val buildDir = layout.buildDirectory.get().asFile.absolutePath
                val podsPath = "cocoapods/synthetic/ios/Pods"
                val tflObjCPath = "$buildDir/$podsPath/TensorFlowLiteObjC"
                val tflCPath = "$buildDir/$podsPath/TensorFlowLiteC"

                linkerOpts("-F$tflCPath")
                linkerOpts("-rpath", tflCPath)
                linkerOpts("-framework", "TensorFlowLiteC")

                linkerOpts("-F$tflObjCPath")
                linkerOpts("-rpath", tflObjCPath)
                linkerOpts("-framework", "TFLTensorFlowLite")
            }
        }
    }

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
            implementation(projects.ktensorflowMoko)
            implementation(projects.ktensorflowTensor)
            implementation(projects.ktensorflowGpu)
            implementation(projects.ktensorflowPipeline)
        }

        commonTest.dependencies {
            implementation(libs.kotlin.test)
        }

        androidInstrumentedTest.dependencies {
            implementation(libs.junit)
            implementation(libs.androidx.test.core)
            implementation(libs.androidx.test.junit)
            implementation(libs.androidx.test.runner)
            implementation(libs.kotlin.test)
        }
    }
}

android {
    namespace = "dev.kursor.ktensorflow.test"
    compileSdk = 35
    defaultConfig {
        minSdk = 24
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
}

val copyIosX64TestResources = tasks.register<Copy>("copyIosX64TestResources") {
    from("src/iosTest/resources")
    into("build/bin/iosX64/debugTest/Contents/Resources")
}

tasks.findByName("iosX64Test")!!.dependsOn(copyIosX64TestResources)

val copyIosSimulatorArm64TestResources = tasks.register<Copy>("copyIosSimulatorArm64TestResources") {
    from("src/iosTest/resources")
    into("build/bin/iosSimulatorArm64/debugTest/Contents/Resources")
}

tasks.findByName("iosSimulatorArm64Test")!!.dependsOn(copyIosSimulatorArm64TestResources)