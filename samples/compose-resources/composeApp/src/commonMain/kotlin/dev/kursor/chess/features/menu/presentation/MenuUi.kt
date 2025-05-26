package dev.kursor.chess.features.menu.presentation

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material.Button
import androidx.compose.material.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import ktensorflow.samples.compose_resources.composeapp.generated.resources.Res
import ktensorflow.samples.compose_resources.composeapp.generated.resources.menu_button_game_ai
import ktensorflow.samples.compose_resources.composeapp.generated.resources.menu_button_game_classic
import org.jetbrains.compose.resources.stringResource

@Composable
fun MenuUi(
    component: MenuComponent,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier
            .fillMaxSize()
            .padding(20.dp),
        verticalArrangement = Arrangement.Center
    ) {
        Button(
            onClick = component::onClassicGameClick,
            modifier = Modifier.fillMaxWidth()
        ) {
            Text(
                text = stringResource(Res.string.menu_button_game_classic)
            )
        }

        Button(
            onClick = component::onAiGameClick,
            modifier = Modifier.fillMaxWidth()
        ) {
            Text(
                text = stringResource(Res.string.menu_button_game_ai)
            )
        }
    }
}