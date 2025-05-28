import com.vanniktech.maven.publish.MavenPublishBaseExtension
import com.vanniktech.maven.publish.SonatypeHost
import org.gradle.api.Plugin
import org.gradle.api.Project
import org.gradle.kotlin.dsl.configure

class PublishingPlugin : Plugin<Project> {

    override fun apply(project: Project): Unit = with(project) {
        pluginManager.apply(libs.plugins.publishing.get().pluginId)

        extensions.configure<MavenPublishBaseExtension> {
            publishToMavenCentral(SonatypeHost.CENTRAL_PORTAL)

            signAllPublications()

            coordinates(group.toString(), project.name, version.toString())

            pom {
                name.set("KTensorFlow")
                description.set("A Kotlin Multiplatform library for TensorFlow.")
                inceptionYear.set("2025")
                url.set("https://github.com/kursor1337/KTensorFlow/")
                licenses {
                    license {
                        name.set("The Apache License, Version 2.0")
                        url.set("https://www.apache.org/licenses/LICENSE-2.0.txt")
                        distribution.set("https://www.apache.org/licenses/LICENSE-2.0.txt")
                    }
                }
                developers {
                    developer {
                        id.set("kursor")
                        name.set("Sergey Kurochkin")
                        url.set("https://github.com/kursor1337/")
                    }
                }
                scm {
                    url.set("https://github.com/kursor1337/KTensorFlow/")
                    connection.set("scm:git:git://github.com/kursor1337/KTensorFlow.git")
                    developerConnection.set("scm:git:ssh://git@github.com/kursor1337/KTensorFlow.git")
                }
            }
        }
    }
}
