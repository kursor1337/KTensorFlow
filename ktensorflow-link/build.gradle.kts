plugins {
    `kotlin-dsl`
    alias(libs.plugins.convention.publishing)
}

dependencies {
    compileOnly(libs.kotlin.multiplatform.gradle.plugin)
    compileOnly(libs.cocoapods.gradle.plugin)

    // Плагин компилируется против KMP/cocoapods как compileOnly, поэтому для тестов
    // эти же зависимости нужны на runtime-класспасе
    testImplementation(libs.kotlin.multiplatform.gradle.plugin)
    testImplementation(libs.cocoapods.gradle.plugin)
    testImplementation(libs.junit)
}

gradlePlugin {
    plugins {
        register("link") {
            id = "dev.kursor.ktensorflow.link"
            implementationClass = "dev.kursor.ktensorflow.link.KTensorFlowLinkPlugin"
        }
    }
}
