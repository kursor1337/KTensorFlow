package dev.kursor.ktensorflow.tensor.impl

import dev.kursor.ktensorflow.tensor.Tensor
import dev.kursor.ktensorflow.tensor.TensorDataType
import dev.kursor.ktensorflow.tensor.TensorShape

internal class TensorImpl(
    override val dataType: TensorDataType,
    override val shape: TensorShape,
    override val data: ByteArray
) : Tensor
