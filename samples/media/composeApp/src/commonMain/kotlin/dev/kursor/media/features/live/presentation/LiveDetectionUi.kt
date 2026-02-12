package dev.kursor.media.features.live.presentation

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.IntrinsicSize
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.wrapContentSize
import androidx.compose.material.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import dev.icerock.moko.permissions.Permission
import dev.icerock.moko.permissions.camera.CAMERA
import dev.icerock.moko.permissions.compose.BindEffect
import dev.icerock.moko.permissions.compose.rememberPermissionsControllerFactory
import dev.kursor.ktensorflow.media.Image
import dev.kursor.ktensorflow.media.ImageTensor
import dev.kursor.ktensorflow.media.PixelFormat
import dev.kursor.ktensorflow.media.camera.LiveCameraUi
import dev.kursor.ktensorflow.media.grayscale
import dev.kursor.ktensorflow.media.resize
import dev.kursor.ktensorflow.media.tensorize
import dev.kursor.ktensorflow.media.toImageTensor
import dev.kursor.ktensorflow.tensor.get
import dev.kursor.ktensorflow.tensor.normalize
import dev.kursor.ktensorflow.tensor.toFloatTensor

@Composable
fun LiveDetectionUi(
    component: LiveDetectionComponent,
    modifier: Modifier = Modifier
) {
    val detectedDigit by component.detectedDigit.collectAsState()
    var frame by remember { mutableStateOf<Image?>(null) }
    val imageTensor = remember(frame) {
        frame
            ?.resize(28, 28)
            ?.grayscale()
            ?.tensorize<Float>()
            ?.normalize()
            ?.toImageTensor(PixelFormat.Grayscale)
    }

    val permissionsControllerFactory = rememberPermissionsControllerFactory()
    val permissionsController = remember(permissionsControllerFactory) {
        permissionsControllerFactory.createPermissionsController()
    }

    BindEffect(permissionsController)

    LaunchedEffect(Unit) {
        if (!permissionsController.isPermissionGranted(Permission.CAMERA)) {
            try {
                permissionsController.providePermission(Permission.CAMERA)
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }
    }

    Box(modifier) {
        LiveCameraUi(
            onFrame = {
                frame = it
                component.onFrame(it)
            },
            modifier = Modifier.fillMaxSize()
        )

        Row(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .padding(bottom = 20.dp)
        ) {
            imageTensor
                ?.let { ImageTensorUi(imageTensor = it) }
                ?: Box(modifier = Modifier.weight(1f))

            Text(
                text = detectedDigit.toString(),
                textAlign = TextAlign.Center,
                fontSize = 72.sp,
                modifier = Modifier.weight(1f)
            )
        }

    }
}

@Composable
private fun ImageTensorUi(
    imageTensor: ImageTensor<Float>,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier
    ) {
        for (y in 0 until imageTensor.height) {
            Row {
                for (x in 0 until imageTensor.width) {
                    Pixel(
                        grayscaleValue = imageTensor[y, x, 0],
                        modifier = Modifier.size(8.dp)
                    )
                }
            }
        }
    }
}

@Composable
private fun Pixel(
    grayscaleValue: Float,
    modifier: Modifier = Modifier
) {
    val value = grayscaleValue.coerceIn(0f..1f)
    Box(
        modifier = modifier
            .background(
                Color(
                    red = value,
                    green = value,
                    blue = value
                )
            )
    )
}