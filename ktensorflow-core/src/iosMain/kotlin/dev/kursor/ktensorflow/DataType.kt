package dev.kursor.ktensorflow

import cocoapods.TensorFlowLiteObjC.TFLTensorDataType

internal fun TFLTensorDataType.toKTensorFlow(): DataType = when (this) {
    TFLTensorDataType.TFLTensorDataTypeFloat32 -> DataType.Float32
    TFLTensorDataType.TFLTensorDataTypeInt32 -> DataType.Int32
    TFLTensorDataType.TFLTensorDataTypeUInt8 -> DataType.UInt8
    TFLTensorDataType.TFLTensorDataTypeInt64 -> DataType.Int64
    TFLTensorDataType.TFLTensorDataTypeInt8 -> DataType.Int8
    TFLTensorDataType.TFLTensorDataTypeInt16 -> DataType.Int16
    TFLTensorDataType.TFLTensorDataTypeFloat16 -> DataType.Float16
    TFLTensorDataType.TFLTensorDataTypeBFloat16 -> DataType.BFloat16
    TFLTensorDataType.TFLTensorDataTypeFloat64 -> DataType.Float64
    TFLTensorDataType.TFLTensorDataTypeBool -> DataType.Bool
    // NoType по документации TensorFlow Lite означает ошибку в самой модели
    else -> throw IllegalArgumentException("Unsupported data type: $this")
}
