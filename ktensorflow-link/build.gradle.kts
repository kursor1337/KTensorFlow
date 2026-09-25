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

// Версия подов TensorFlowLiteObjC приходит из каталога версий, а не пишется в код плагина:
// плагин обязан линковать у пользователя ровно ту версию, против которой собран cinterop
// в ktensorflow-core, и раньше она была продублирована в восьми местах
val generateVersions = tasks.register("generateVersions") {
    val version = libs.versions.tensorflow.ios.get()
    val outputDir = layout.buildDirectory.dir("generated/sources/versions/kotlin")
    inputs.property("version", version)
    outputs.dir(outputDir)
    doLast {
        val file = outputDir.get().file("dev/kursor/ktensorflow/link/Versions.kt").asFile
        file.parentFile.mkdirs()
        file.writeText(
            "package dev.kursor.ktensorflow.link\n\n" +
                "/** Version of the TensorFlowLiteObjC pods the library is built against. */\n" +
                "internal const val TENSORFLOW_LITE_OBJC_VERSION = \"$version\"\n"
        )
    }
}

kotlin.sourceSets.main {
    kotlin.srcDir(generateVersions)
}
