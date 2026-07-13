package dev.kursor.vision.features.live.presentation

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
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
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import dev.icerock.moko.permissions.Permission
import dev.icerock.moko.permissions.camera.CAMERA
import dev.icerock.moko.permissions.compose.BindEffect
import dev.icerock.moko.permissions.compose.rememberPermissionsControllerFactory
import dev.kursor.ktensorflow.vision.Image
import dev.kursor.ktensorflow.vision.ImageTensor
import dev.kursor.ktensorflow.vision.ImageTensorLayout
import dev.kursor.ktensorflow.vision.PixelFormat
import dev.kursor.vision.core.camera.LiveCameraUi
import dev.kursor.ktensorflow.vision.grayscale
import dev.kursor.ktensorflow.vision.resize
import dev.kursor.ktensorflow.vision.tensorize
import dev.kursor.ktensorflow.vision.toImageTensor
import dev.kursor.ktensorflow.tensor.normalize
import dev.kursor.ktensorflow.vision.resizeWithPad

@Composable
fun LiveDetectionUi(
    component: LiveDetectionComponent,
    modifier: Modifier = Modifier
) {
    val detectionResult by component.detectionResults.collectAsState()
    var frame by remember { mutableStateOf<Image?>(null) }
    val imageTensor = remember(frame) {
        frame
            ?.resizeWithPad(320, 320)
            ?.tensorize<UByte>()
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

        // Добавленный Canvas для отрисовки Bounding Boxes поверх камеры
        Canvas(modifier = Modifier.fillMaxSize()) {
            detectionResult?.objects?.forEach { detectedObject ->
                val box = detectedObject.boundingBox
                val left = box.left.toFloat()
                val top = box.top.toFloat()
                val width = (box.right - box.left).toFloat()
                val height = (box.bottom - box.top).toFloat()

                // Отрисовка прямоугольника
                drawRect(
                    color = Color.Green,
                    topLeft = Offset(x = left, y = top),
                    size = Size(width = width, height = height),
                    style = Stroke(width = 3.dp.toPx()) // Толщина линии рамки
                )
            }
        }

        Row(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .padding(bottom = 20.dp)
        ) {
            imageTensor
                ?.let { ImageTensorUi(imageTensor = it) }
                ?: Box(modifier = Modifier.weight(1f))
        }
    }
}

@Composable
private fun ImageTensorUi(
    imageTensor: ImageTensor<UByte>,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier
    ) {
        for (y in 0 until imageTensor.height) {
            Row {
                for (x in 0 until imageTensor.width) {
                    Pixel(
                        redValue = imageTensor[0, y, x, 0],
                        greenValue = imageTensor[0, y, x, 1],
                        blueValue = imageTensor[0, y, x, 2],
                        modifier = Modifier.size(0.5.dp)
                    )
                }
            }
        }
    }
}

@Composable
private fun Pixel(
    redValue: UByte,
    greenValue: UByte,
    blueValue: UByte,
    modifier: Modifier = Modifier
) {
    val red = redValue.toFloat() / 255
    val green = greenValue.toFloat() / 255
    val blue = blueValue.toFloat() / 255
    Box(
        modifier = modifier
            .background(
                Color(
                    red = red,
                    green = green,
                    blue = blue
                )
            )
    )
}