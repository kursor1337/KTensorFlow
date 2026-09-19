import kotlinx.validation.ApiValidationExtension
import kotlinx.validation.ExperimentalBCVApi
import org.gradle.api.Plugin
import org.gradle.api.Project
import org.gradle.kotlin.dsl.configure

class BinaryCompatibilityPlugin : Plugin<Project> {

    @OptIn(ExperimentalBCVApi::class)
    override fun apply(project: Project): Unit = with(project) {
        pluginManager.apply(libs.plugins.binary.compatibility.validator.get().pluginId)

        extensions.configure<ApiValidationExtension> {
            klib {
                enabled = true
            }
        }
    }
}
