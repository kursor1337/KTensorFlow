package dev.kursor.media

import android.app.Application
import dev.kursor.media.core.SharedApplicationProvider

val Application.sharedApplication get() = (this as SharedApplicationProvider).sharedApplication