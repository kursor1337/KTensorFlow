package dev.kursor.ktensorflow.moko

import dev.icerock.moko.resources.AssetResource
import dev.icerock.moko.resources.FileResource
import dev.kursor.ktensorflow.api.ModelDesc

expect fun ModelDesc.Companion.FileResource(resource: FileResource): ModelDesc

expect fun ModelDesc.Companion.AssetResource(resource: AssetResource): ModelDesc