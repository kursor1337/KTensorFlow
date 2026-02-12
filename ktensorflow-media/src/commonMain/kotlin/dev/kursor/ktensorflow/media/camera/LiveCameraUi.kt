package dev.kursor.ktensorflow.media.camera

import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import dev.kursor.ktensorflow.media.Image

@Composable
expect fun LiveCameraUi(
    modifier: Modifier = Modifier,
    onFrame: (Image) -> Unit
)
