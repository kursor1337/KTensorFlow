package dev.kursor.vision.features.live

import com.arkivanov.decompose.ComponentContext
import dev.kursor.vision.core.ComponentFactory
import dev.kursor.vision.features.live.data.LiveDetectionRepository
import dev.kursor.vision.features.live.data.LiveDetectionRepositoryImpl
import dev.kursor.vision.features.live.presentation.LiveDetectionComponent
import dev.kursor.vision.features.live.presentation.RealLiveDetectionComponent
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