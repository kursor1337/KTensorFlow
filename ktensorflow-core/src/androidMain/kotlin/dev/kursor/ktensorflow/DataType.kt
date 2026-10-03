package dev.kursor.ktensorflow

import org.tensorflow.lite.DataType as TFLDataType

internal fun TFLDataType.toKTensorFlow(): DataType = when (this) {
    TFLDataType.FLOAT32 -> DataType.Float32
    TFLDataType.INT32 -> DataType.Int32
    TFLDataType.UINT8 -> DataType.UInt8
    TFLDataType.INT64 -> DataType.Int64
    TFLDataType.INT8 -> DataType.Int8
    TFLDataType.INT16 -> DataType.Int16
    TFLDataType.BOOL -> DataType.Bool
    TFLDataType.STRING -> DataType.String
}