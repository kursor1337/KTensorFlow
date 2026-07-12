package dev.kursor.media

import androidx.compose.material.MaterialTheme
import androidx.compose.ui.window.ComposeUIViewController
import dev.kursor.media.features.root.presentation.RootComponent
import dev.kursor.media.features.root.presentation.RootUi

fun MainViewController(
    rootComponent: RootComponent
) = ComposeUIViewController {
    MaterialTheme {
        RootUi(rootComponent)
    }
}