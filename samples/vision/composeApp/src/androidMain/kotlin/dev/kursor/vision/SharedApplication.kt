package dev.kursor.vision

import android.app.Application
import dev.kursor.vision.core.SharedApplicationProvider

val Application.sharedApplication get() = (this as SharedApplicationProvider).sharedApplication