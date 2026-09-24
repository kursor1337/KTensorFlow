import org.gradle.api.Plugin
import org.gradle.api.Project
import org.jetbrains.kotlin.gradle.dsl.KotlinMultiplatformExtension
import org.jetbrains.kotlin.gradle.dsl.abi.ExperimentalAbiValidation

/**
 * Проверка публичного ABI модуля встроенными средствами Kotlin Gradle Plugin.
 *
 * Отдельный плагин binary-compatibility-validator переведён в maintenance mode и не видит
 * Android-таргет плагина com.android.kotlin.multiplatform.library: он берёт только компиляцию
 * `release` старого androidTarget(), а у нового таргета она одна и называется `main`. Из-за
 * этого Android-API не проверялся вовсе. Встроенная валидация поддерживает этот таргет начиная
 * с Kotlin 2.4.20 (KT-85950) и проверяет Android-дамп вместе с klib-дампом для iOS.
 *
 * Задачи: `checkKotlinAbi` (входит в `check` и в гейт публикации) и `updateKotlinAbi`.
 */
class BinaryCompatibilityPlugin : Plugin<Project> {

    @OptIn(ExperimentalAbiValidation::class)
    override fun apply(project: Project): Unit = with(project) {
        pluginManager.withPlugin("org.jetbrains.kotlin.multiplatform") {
            // Class-вариант getByType, а не reified из kotlin-dsl: импорт reified-расширения
            // IDE считает неиспользуемым и вырезает при оптимизации импортов, и сборка ломается
            extensions.getByType(KotlinMultiplatformExtension::class.java).abiValidation {
            }
        }
    }
}
