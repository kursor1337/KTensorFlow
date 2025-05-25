package dev.kursor.chess

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.systemBarsPadding
import androidx.compose.material.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import com.arkivanov.decompose.retainedComponent
import dev.kursor.chess.features.root.presentation.RootUi
import org.jetbrains.compose.resources.ExperimentalResourceApi

class MainActivity : ComponentActivity() {
    @OptIn(ExperimentalResourceApi::class)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val rootComponent = retainedComponent { componentContext ->
            application.sharedApplication.createRootComponent(componentContext)
        }

        setContent {
            MaterialTheme {
                RootUi(
                    component = rootComponent,
                    modifier = Modifier.systemBarsPadding()
                )
            }
        }
    }
}

@Preview
@Composable
fun AppAndroidPreview() {
    App()
}