import io.gitlab.arturbosch.detekt.Detekt
import io.gitlab.arturbosch.detekt.extensions.DetektExtension
import org.gradle.api.Plugin
import org.gradle.api.Project
import org.gradle.kotlin.dsl.configure
import org.gradle.kotlin.dsl.dependencies
import org.gradle.kotlin.dsl.withType

class DetektPlugin : Plugin<Project> {

    override fun apply(target: Project) = with(target) {
        pluginManager.apply(libs.plugins.detekt.get().pluginId)

        extensions.configure<DetektExtension> {
            toolVersion = libs.versions.detekt.get()
            source.setFrom(files(rootDir))
            parallel = true
            ignoreFailures = false
            disableDefaultRuleSets = true
        }

        dependencies {
            add("detektPlugins", libs.detekt.formatting)
        }

        tasks.withType<Detekt>().configureEach {
            exclude(
                "**/build/**",
                "**/resources/**"
            )
            reports {
                xml {
                    outputLocation.set(file("build/reports/detekt-results.xml"))
                    required.set(true)
                }
                html {
                    outputLocation.set(file("build/reports/detekt-results.html"))
                    required.set(true)
                }
                txt.required.set(false)
                md.required.set(false)
                sarif.required.set(false)
            }
        }
    }
}
