package dev.kursor.vision.features.live.presentation

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.size
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import dev.icerock.moko.permissions.Permission
import dev.icerock.moko.permissions.camera.CAMERA
import dev.icerock.moko.permissions.compose.BindEffect
import dev.icerock.moko.permissions.compose.rememberPermissionsControllerFactory
import dev.kursor.ktensorflow.vision.ImageTensor
import dev.kursor.ktensorflow.vision.scaleForContainer
import dev.kursor.vision.core.camera.LiveCameraUi
import dev.kursor.vision.core.utils.toComposeRect
import kotlin.math.roundToInt

@Composable
fun LiveDetectionUi(
    component: LiveDetectionComponent,
    modifier: Modifier = Modifier
) {
    val detectionResult by component.detectionResults.collectAsState()

    val permissionsControllerFactory = rememberPermissionsControllerFactory()
    val permissionsController = remember(permissionsControllerFactory) {
        permissionsControllerFactory.createPermissionsController()
    }

    val textMeasurer = rememberTextMeasurer()

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

    BoxWithConstraints(modifier) {
        LiveCameraUi(
            onFrame = component::onFrame,
            modifier = Modifier.fillMaxSize()
        )

        Canvas(modifier = Modifier.fillMaxSize()) {
            val canvasWidth = size.width
            val canvasHeight = size.height

            detectionResult.objects.forEach { detectedObject ->
                val box = detectedObject
                    .boundingBox
                    .scaleForContainer(
                        originalContainerWidth = detectedObject.imageWidth,
                        originalContainerHeight = detectedObject.imageHeight,
                        containerWidth = canvasWidth,
                        containerHeight = canvasHeight
                    )
                    .toComposeRect()

                val strokeWidth = 3.dp.toPx()

                drawRect(
                    color = Color.Green,
                    topLeft = box.topLeft,
                    size = box.size,
                    style = Stroke(width = strokeWidth)
                )

                val confidencePercent = (detectedObject.confidence * 100).roundToInt()
                val text = "${detectedObject.label} $confidencePercent%"

                val textLayoutResult = textMeasurer.measure(
                    text = text,
                    style = TextStyle(
                        color = Color.White,
                        fontSize = 14.sp,
                        fontWeight = FontWeight.Bold
                    )
                )

                val textPadding = 4.dp.toPx()
                val bgWidth = textLayoutResult.size.width + (textPadding * 2)
                val bgHeight = textLayoutResult.size.height + (textPadding * 2)

                var textTopY = box.top - bgHeight - (strokeWidth / 2)

                if (textTopY < 0f) {
                    textTopY = box.top + (strokeWidth / 2)
                }

                if (textTopY + bgHeight > canvasHeight) {
                    textTopY = canvasHeight - bgHeight
                }

                var textLeftX = box.left

                if (textLeftX + bgWidth > canvasWidth) {
                    textLeftX = canvasWidth - bgWidth
                }

                if (textLeftX < 0f) {
                    textLeftX = 0f
                }

                drawRect(
                    color = Color.Green,
                    topLeft = Offset(x = textLeftX, y = textTopY),
                    size = Size(width = bgWidth, height = bgHeight)
                )

                drawText(
                    textLayoutResult = textLayoutResult,
                    topLeft = Offset(
                        x = textLeftX + textPadding,
                        y = textTopY + textPadding
                    )
                )
            }
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