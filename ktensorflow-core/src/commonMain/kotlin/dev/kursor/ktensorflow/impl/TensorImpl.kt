package dev.kursor.ktensorflow.impl

import dev.kursor.ktensorflow.Tensor
import dev.kursor.ktensorflow.TensorDataType
import dev.kursor.ktensorflow.TensorShape

internal class TensorImpl(
    override val dataType: TensorDataType,
    override val shape: TensorShape,
    override val data: ByteArray
) : Tensor
