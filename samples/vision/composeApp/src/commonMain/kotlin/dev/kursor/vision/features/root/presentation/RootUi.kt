package dev.kursor.vision.features.root.presentation

import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import com.arkivanov.decompose.extensions.compose.stack.Children
import dev.kursor.vision.features.live.presentation.LiveDetectionUi
import dev.kursor.vision.features.menu.presentation.MenuUi

@Composable
fun RootUi(
    component: RootComponent,
    modifier: Modifier = Modifier
) {
    val childStack by component.childStack.collectAsState()
    Children(stack = childStack, modifier = modifier) { child ->
        when (val instance = child.instance) {
            is RootComponent.Child.Menu -> MenuUi(instance.component)
            is RootComponent.Child.LiveDetection -> LiveDetectionUi(instance.component)
        }
    }
}