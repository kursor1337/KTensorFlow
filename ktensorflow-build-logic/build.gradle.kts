plugins {
    `kotlin-dsl`
}

repositories {
    google()
    gradlePluginPortal()
    mavenCentral()
}

dependencies {
    implementation(libs.detekt.gradle.plugin)
    implementation(libs.publishing.gradle.plugin)
    // compileOnly: в рантайме конвенционный плагин получает тот же KGP, которым модули
    // применяют kotlin-multiplatform, а не вторую его копию в своём classloader'е
    compileOnly(libs.kotlin.multiplatform.gradle.plugin)

    // Workaround for version catalog working inside precompiled scripts
    // Issue - https://github.com/gradle/gradle/issues/15383
    compileOnly(files(libs.javaClass.superclass.protectionDomain.codeSource.location))
}

gradlePlugin {
    plugins {
        register("detekt") {
            id = "convention.detekt"
            implementationClass = "DetektPlugin"
        }
        register("publishing") {
            id = "convention.publishing"
            implementationClass = "PublishingPlugin"
        }
        register("verify") {
            id = "convention.verify"
            implementationClass = "VerificationPlugin"
        }
        register("binaryCompatibility") {
            id = "convention.binaryCompatibility"
            implementationClass = "BinaryCompatibilityPlugin"
        }
    }
}
