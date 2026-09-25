package dev.kursor.vision.core.utils

import androidx.compose.ui.geometry.Rect as ComposeRect
import dev.kursor.ktensorflow.vision.Rect as VisionRect

fun VisionRect.toComposeRect() = ComposeRect(
    left = left.toFloat(),
    top = top.toFloat(),
    right = right.toFloat(),
    bottom = bottom.toFloat()
)