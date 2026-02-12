package dev.kursor.media.features.live

import com.arkivanov.decompose.ComponentContext
import dev.kursor.media.core.ComponentFactory
import dev.kursor.media.features.live.data.LiveDetectionRepository
import dev.kursor.media.features.live.data.LiveDetectionRepositoryImpl
import dev.kursor.media.features.live.presentation.LiveDetectionComponent
import dev.kursor.media.features.live.presentation.RealLiveDetectionComponent
import org.koin.core.component.get
import org.koin.dsl.module

val liveDetectionModule = module {
    single<LiveDetectionRepository> { LiveDetectionRepositoryImpl() }
}

fun ComponentFactory.createLiveDetectionComponent(
    componentContext: ComponentContext
): LiveDetectionComponent {
    return RealLiveDetectionComponent(
        componentContext,
        get()
    )
}