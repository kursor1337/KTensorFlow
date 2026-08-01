import org.jetbrains.kotlin.gradle.dsl.JvmTarget
import org.jetbrains.kotlin.gradle.plugin.mpp.KotlinNativeTarget

plugins {
    alias(libs.plugins.kotlin.multiplatform)
    alias(libs.plugins.kotlin.cocoapods)
    alias(libs.plugins.android.library)
}

kotlin {
    android {
        namespace = "dev.kursor.ktensorflow.test"
        compileSdk = libs.versions.android.compileSdk.get().toInt()
        minSdk = libs.versions.android.minSdk.get().toInt()
        compilerOptions {
            jvmTarget.set(JvmTarget.JVM_11)
        }
    }

    val iosConfigure: KotlinNativeTarget.() -> Unit = {
        binaries {
            getTest("DEBUG").apply {
                val buildDir = layout.buildDirectory.get().asFile.absolutePath
                val podsPath = "cocoapods/synthetic/ios/Pods"
                val tflObjCPath = "$buildDir/$podsPath/TensorFlowLiteObjC"
                val tflCPath = "$buildDir/$podsPath/TensorFlowLiteC/Frameworks"

                linkerOpts("-F$tflCPath")
                linkerOpts("-rpath", tflCPath)
                linkerOpts("-framework", "TensorFlowLiteC")

                linkerOpts("-F$tflCPath")
                linkerOpts("-rpath", tflCPath)
                linkerOpts("-framework", "TensorFlowLiteCMetal")

                linkerOpts("-F$tflCPath")
                linkerOpts("-rpath", tflCPath)
                linkerOpts("-framework", "TensorFlowLiteCCoreML")

                linkerOpts("-F$tflObjCPath")
                linkerOpts("-rpath", tflObjCPath)
                linkerOpts("-framework", "TFLTensorFlowLite")
            }
        }
    }

    iosArm64(iosConfigure)
    iosSimulatorArm64(iosConfigure)

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
            linkOnly = true
        }
        pod("TensorFlowLiteObjC/Metal") {
            moduleName = "TFLTensorFlowLite"
            version = "2.17.0"
            linkOnly = true
        }
        pod("TensorFlowLiteObjC/CoreML") {
            moduleName = "TFLTensorFlowLite"
            version = "2.17.0"
            linkOnly = true
        }
    }

    sourceSets {
        commonMain.dependencies {
            implementation(projects.ktensorflowCore)
            implementation(projects.ktensorflowMoko)
            implementation(projects.ktensorflowTensor)
            implementation(projects.ktensorflowGpu)
            implementation(projects.ktensorflowNpu)
            implementation(projects.ktensorflowPipeline)
            implementation(projects.ktensorflowVision)
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

val copyIosSimulatorArm64TestResources = tasks.register<Copy>("copyIosSimulatorArm64TestResources") {
    from("src/iosTest/resources")
    into("build/bin/iosSimulatorArm64/debugTest/Contents/Resources")
}

tasks.findByName("iosSimulatorArm64Test")!!.dependsOn(copyIosSimulatorArm64TestResources)

val startEmulator = tasks.register<Exec>("startEmulator") {
    group = "verification"
    description = "Starts the first available Android emulator if no emulator is currently running (ignores physical devices)"

    commandLine("sh", "-c", """
        set -e

        # Get the first available AVD name
        AVD_NAME=$(emulator -list-avds | head -n 1)

        if [ -z "${"$"}AVD_NAME" ]; then
          echo "❌ No Android Virtual Devices found. Create one with 'avdmanager create avd'."
          exit 1
        fi

        # Check if an emulator (not physical device) is already connected
        EMULATOR_COUNT=$(adb devices | grep 'emulator-[0-9]\+' | grep -w "device" | wc -l | tr -d ' ')

        if [ "${"$"}EMULATOR_COUNT" -eq 0 ]; then
          echo "🚀 Starting emulator: ${"$"}AVD_NAME ..."
          nohup emulator -avd "${"$"}AVD_NAME" -no-snapshot -no-boot-anim > /dev/null 2>&1 &

          echo "⏳ Waiting for emulator to boot..."
          adb wait-for-device
          adb shell 'until [[ $(getprop sys.boot_completed) -eq 1 ]]; do sleep 1; done'
          echo "✅ Emulator ${"$"}AVD_NAME is ready."
        else
          echo "📱 Emulator already running — skipping startup."
        fi
    """.trimIndent())
}

tasks.matching { it.name.startsWith("connected") && it.name.endsWith("AndroidTest") }.configureEach {
    dependsOn(startEmulator)
}

tasks.register("runAllTests") {
    group = "verification"
    description = "Runs all common, Android, and iOS tests"
    dependsOn(
        "allTests",
        "iosSimulatorArm64Test",
        "connectedAndroidTest"
    )
}
