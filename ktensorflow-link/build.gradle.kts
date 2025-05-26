plugins {
    `kotlin-dsl`
}

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
