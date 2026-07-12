package dev.kursor.media.core

import co.touchlab.kermit.Logger
import co.touchlab.kermit.Severity
import com.arkivanov.decompose.ComponentContext
import dev.kursor.media.core.configuration.BuildType
import dev.kursor.media.core.configuration.Configuration
import dev.kursor.media.features.allFeatureModules
import dev.kursor.media.features.root.createRootComponent
import dev.kursor.media.features.root.presentation.RootComponent
import org.koin.core.Koin

class SharedApplication(configuration: Configuration) {

    private val koin: Koin

    init {
        if (configuration.buildType == BuildType.Release) {
            Logger.setMinSeverity(Severity.Assert)
        }
        koin = createKoin(configuration)
    }

    fun createRootComponent(
        componentContext: ComponentContext
    ): RootComponent {
        return koin.get<ComponentFactory>().createRootComponent(componentContext)
    }

    internal inline fun <reified T : Any> get(): T = koin.get<T>()

    private fun createKoin(configuration: Configuration): Koin {
        return Koin().apply {
            loadModules(
                listOf(
                    commonCoreModule(configuration),
                    platformCoreModule(configuration),
                ) + allFeatureModules
            )
            declare(ComponentFactory(this))
            createEagerInstances()
        }
    }
}

interface SharedApplicationProvider {
    val sharedApplication: SharedApplication
}