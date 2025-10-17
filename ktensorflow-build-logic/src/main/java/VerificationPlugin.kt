import org.gradle.api.GradleException
import org.gradle.api.Plugin
import org.gradle.api.Project
import org.gradle.api.publish.maven.tasks.PublishToMavenRepository

private val modulesNeededToBePublished = setOf(
    "ktensorflow-core",
    "ktensorflow-gpu",
    "ktensorflow-npu",
    "ktensorflow-link",
    "ktensorflow-moko",
    "ktensorflow-compose",
    "ktensorflow-pipeline",
    "ktensorflow-tensor"
)

class VerificationPlugin : Plugin<Project> {
    override fun apply(project: Project): Unit = with(project) {

        val modulesToBePublished = provider {
            rootProject
                .childProjects
                .values
                .filter {
                    it.pluginManager.hasPlugin("convention.publishing")
                }
                .map { it.name }
                .toSet()
        }
        // Register verification task
        val verifyModulesTask = tasks.register("verifyModules") {
            group = "verification"
            description = "Checks if modules to be published are correct"

            doLast {
                if (modulesNeededToBePublished != modulesToBePublished.get()) {
                    throw GradleException(
                        """
                        Modules to be published are not correct:

                        Expected: $modulesNeededToBePublished
                        Actual: $modulesToBePublished
                        """.trimIndent()
                    )
                }
            }
        }

        // ✅ After everything is configured
        gradle.projectsEvaluated {
            // point to the real runAllTests task
            val runAllTestsPath = ":ktensorflow-test:runAllTests"
            val runAllTests = gradle.rootProject.tasks.findByPath(runAllTestsPath)
                ?: throw GradleException("Task '$runAllTestsPath' not found — make sure ktensorflow-test defines it")

            val apiCheckPath = "apiCheck"
            val apiCheck = gradle.rootProject.tasks.findByPath(apiCheckPath)
                ?: throw GradleException("Task '$apiCheckPath' not found — make sure ktensorflow-test defines it")

            // attach verification & tests to publishing tasks in all subprojects
            rootProject.allprojects.forEach { sub ->
                sub.tasks.matching { it.name == "publishToMavenCentral" }.configureEach {
                    dependsOn(runAllTests)
                    dependsOn(verifyModulesTask)
                    dependsOn(apiCheck)
                    println("✔ Linked $runAllTestsPath and :verifyModules to ${sub.path}:$name")
                }
            }
        }
    }
}
