enableFeaturePreview("TYPESAFE_PROJECT_ACCESSORS")
pluginManagement {
    includeBuild("ktensorflow-link")
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

include(":samples:compose-resources:chess-engine")
include(":samples:compose-resources:composeApp")
