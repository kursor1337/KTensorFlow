package dev.kursor.ktensorflow.moko

import android.content.Context
import android.content.res.Resources
import dev.icerock.moko.resources.AssetResource
import dev.icerock.moko.resources.FileResource
import dev.kursor.ktensorflow.ModelDesc
import java.io.File

actual fun ModelDesc.Companion.FileResource(resource: FileResource): ModelDesc {
    return ModelDesc.File(resource.openAsFile(appContext))
}

actual fun ModelDesc.Companion.AssetResource(resource: AssetResource): ModelDesc {
    return ModelDesc.File(resource.openAsFile(appContext))
}

private fun FileResource.openAsFile(context: Context): File {
    val resources: Resources = context.resources
    return resources.openRawResource(rawResId).use { inputStream ->
        val tmpFile = File.createTempFile("prefix", "suffix", context.cacheDir)
        tmpFile.outputStream().use {
            inputStream.copyTo(it)
        }
        tmpFile
    }
}

private fun AssetResource.openAsFile(context: Context): File {
    val inputStream = context.assets.open(path)
    return inputStream.use { inputStream ->
        val tmpFile = File.createTempFile("prefix", "suffix", context.cacheDir)
        tmpFile.outputStream().use {
            inputStream.copyTo(it)
        }
        tmpFile
    }
}
