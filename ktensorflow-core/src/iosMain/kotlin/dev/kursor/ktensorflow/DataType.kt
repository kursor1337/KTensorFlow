package dev.kursor.ktensorflow

import cocoapods.TensorFlowLiteObjC.TFLTensorDataType

fun TFLTensorDataType.toKTensorFlow(): DataType = when (this) {
    TFLTensorDataType.TFLTensorDataTypeFloat32 -> DataType.Float32
    TFLTensorDataType.TFLTensorDataTypeInt32 -> DataType.Int32
    TFLTensorDataType.TFLTensorDataTypeUInt8 -> DataType.UInt8
    TFLTensorDataType.TFLTensorDataTypeInt64 -> DataType.Int64
    else -> throw IllegalArgumentException("Unsupported data type: $this")
}
