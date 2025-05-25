package dev.kursor.chess

import android.app.Application
import dev.kursor.chess.core.SharedApplicationProvider

val Application.sharedApplication get() = (this as SharedApplicationProvider).sharedApplication