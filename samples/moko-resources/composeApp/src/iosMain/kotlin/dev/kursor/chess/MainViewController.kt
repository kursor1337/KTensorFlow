package dev.kursor.chess

import androidx.compose.material.MaterialTheme
import androidx.compose.ui.window.ComposeUIViewController
import dev.kursor.chess.features.root.presentation.RootComponent
import dev.kursor.chess.features.root.presentation.RootUi

fun MainViewController(
    rootComponent: RootComponent
) = ComposeUIViewController {
    MaterialTheme {
        RootUi(rootComponent)
    }
}