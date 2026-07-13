enableFeaturePreview("TYPESAFE_PROJECT_ACCESSORS")
pluginManagement {
    includeBuild("ktensorflow-build-logic")
    // includeBuild("ktensorflow-link")
    repositories {
        google()
        gradlePluginPortal()
        mavenCentral()
    }
}

dependencyResolutionManagement {
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "KTensorFlow"
include(":ktensorflow-core")
include(":ktensorflow-gpu")
include(":ktensorflow-npu")
include(":ktensorflow-link")
include(":ktensorflow-moko")
include(":ktensorflow-compose")
include(":ktensorflow-pipeline")
include(":ktensorflow-tensor")
include(":ktensorflow-vision")
include(":ktensorflow-test")
include(":samples:chess-engine")
include(":samples:compose-resources:composeApp")
include(":samples:moko-resources:composeApp")
include(":samples:vision:composeApp")
