package dev.kursor.ktensorflow.moko

import dev.icerock.moko.resources.AssetResource
import dev.icerock.moko.resources.FileResource
import dev.kursor.ktensorflow.ModelDesc

actual fun ModelDesc.Companion.FileResource(resource: FileResource): ModelDesc {
    return ModelDesc.PathInBundle(resource.path)
}

actual fun ModelDesc.Companion.AssetResource(resource: AssetResource): ModelDesc {
    return ModelDesc.PathInBundle(resource.path)
}