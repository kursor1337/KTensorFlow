plugins {
    `kotlin-dsl`
}

group = "io.github.kursor1337"
version = "0.1"

dependencies {
    compileOnly(libs.kotlin.multiplatform.gradle.plugin)
    compileOnly(libs.cocoapods.gradle.plugin)
}

gradlePlugin {
    plugins {
        register("link") {
            id = "dev.kursor.ktensorflow.link"
            implementationClass = "dev.kursor.ktensorflow.link.KTensorFlowLinkPlugin"
        }
    }
}
