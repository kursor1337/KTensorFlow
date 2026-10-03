package dev.kursor.vision

import androidx.compose.material.MaterialTheme
import androidx.compose.ui.window.ComposeUIViewController
import dev.kursor.vision.features.root.presentation.RootComponent
import dev.kursor.vision.features.root.presentation.RootUi

fun MainViewController(
    rootComponent: RootComponent
) = ComposeUIViewController {
    MaterialTheme {
        RootUi(rootComponent)
    }
}