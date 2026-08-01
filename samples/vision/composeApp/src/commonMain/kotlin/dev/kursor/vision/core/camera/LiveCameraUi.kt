package dev.kursor.vision.core.camera

import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import dev.kursor.ktensorflow.vision.Image

@Composable
expect fun LiveCameraUi(
    modifier: Modifier = Modifier,
    onFrame: (Image) -> Unit
)
