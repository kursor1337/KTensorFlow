package dev.kursor.chess.features.root.presentation

import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import com.arkivanov.decompose.extensions.compose.stack.Children
import dev.kursor.chess.features.game.presentation.GameUi

@Composable
fun RootUi(
    component: RootComponent,
    modifier: Modifier = Modifier
) {
    val childStack by component.childStack.collectAsState()
    Children(stack = childStack, modifier = modifier) { child ->
        when (val instance = child.instance) {
            is RootComponent.Child.Game -> GameUi(instance.component)
        }
    }
}