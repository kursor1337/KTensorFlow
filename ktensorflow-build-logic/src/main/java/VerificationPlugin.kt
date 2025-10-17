import org.gradle.api.GradleException
import org.gradle.api.Plugin
import org.gradle.api.Project

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

private val modulesToApiCheck =
    modulesNeededToBePublished - "ktensorflow-link"

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

        project.tasks.forEach {
            println(it.name)
        }
        gradle.afterProject {
            val runAllTestsPath = ":ktensorflow-test:runAllTests"
            val runAllTests = rootProject.tasks.findByPath(runAllTestsPath)
                ?: throw GradleException("Task '$runAllTestsPath' not found — make sure ktensorflow-test defines it")

            val apiCheckTasks = modulesToApiCheck
                .map { ":$it:apiCheck" }
                .map { rootProject.tasks.findByPath(it) }

            rootProject.allprojects.forEach { sub ->
                sub.tasks.matching { it.name == "publishToMavenCentral" }.configureEach {
                    dependsOn(runAllTests)
                    dependsOn(verifyModulesTask)
                    apiCheckTasks.forEach { dependsOn(it!!) }
                    println("✔ Linked $runAllTestsPath and :verifyModules to ${sub.path}:$name")
                }
            }
        }
    }
}
