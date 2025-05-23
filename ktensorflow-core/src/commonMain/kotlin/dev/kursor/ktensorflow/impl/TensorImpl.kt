package dev.kursor.ktensorflow.impl

import dev.kursor.ktensorflow.api.Tensor
import dev.kursor.ktensorflow.api.TensorDataType
import dev.kursor.ktensorflow.api.TensorShape

internal class TensorImpl(
    override val dataType: TensorDataType,
    override val shape: TensorShape,
    override val data: ByteArray
) : Tensor