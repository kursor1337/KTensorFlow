# Migrating from 1.x to 2.x

KTensorFlow 2.0 adds the `ktensorflow-vision` and `ktensorflow-coroutines` modules, zero-copy tensor views, model metadata and a large number of fixes. Most 1.x code keeps compiling, but several APIs changed in ways the compiler or your tests will point out. This guide lists everything you may need to change, module by module.

## Requirements

| | 1.x | 2.0 |
|---|---|---|
| Kotlin | 2.2.20 | 2.4.20 |
| LiteRT (Android) | 1.4.0 | 1.4.2 |
| iOS | 13.0 | 15.0 |
| Android `minSdk` | 24 | 24 |
| App `compileSdk` | no requirement | 34 or newer |
| Linking plugin `dev.kursor.ktensorflow.link` | 0.3 – 1.2 | 2.0 |

Update the linking plugin together with the libraries: it links the TensorFlowLiteObjC pods of exactly the version the library is built against.

```kotlin
plugins {
  id("dev.kursor.ktensorflow.link") version "2.0"
}
```

## Core

### Every failure is a `TensorFlowException`

Loading a model, applying a delegate, resizing an input and running inference now fail with `dev.kursor.ktensorflow.TensorFlowException` on both platforms. On Android 1.x let LiteRT's own `IllegalArgumentException`, `IllegalStateException` or `IOException` through. The platform exception is kept as `cause`.

```kotlin
// 1.x
try { Interpreter(modelDesc, options) } catch (e: IllegalArgumentException) { /* ... */ }

// 2.0
try { Interpreter(modelDesc, options) } catch (e: TensorFlowException) { /* ... */ }
```

### `Interpreter` is `AutoCloseable` and safe to close at any time

`Interpreter` now extends `AutoCloseable`, so `use { }` works. Calls from several threads are serialized by the interpreter, `close()` waits for a running inference instead of releasing native memory under it, and any call after `close()` throws `TensorFlowException`.

`Interpreter` also gained `inputTensorCount`, `outputTensorCount`, `getModelMeta()` and `resizeInput(index, dims)`. `run` now requires every model input and output buffers at least as large as their tensors; a mismatch throws `TensorFlowException`.

### Delegates are `AutoCloseable`

`Delegate` now extends `AutoCloseable` and releases its native resources in `close()`. Close a delegate only after every interpreter that uses it has been closed:

```kotlin
val gpu = GpuDelegate()
val interpreter = Interpreter(modelDesc, InterpreterOptions(numThreads = 4, useXNNPACK = true, delegates = listOf(gpu)))
// ...
interpreter.close()
gpu.close()
```

If you implement `Delegate` yourself, add a `close()` implementation.

Every available delegate in `InterpreterOptions(delegates = ...)` is applied, in list order, on both platforms. A delegate that is available on the device but cannot be applied to the model makes interpreter creation fail with `TensorFlowException`; see the README for a CPU fallback.

### Library interfaces are closed to implementation

`Interpreter`, `Tensor`, `PhysicalTensor`, `TensorView` and `ImageTensor` are annotated with `@SubclassOptInRequired(InternalKTensorFlowApi::class)`. Using them is unchanged; implementing them outside the library now requires an explicit opt-in, because the library adds members to them. Fakes in tests need `@OptIn(InternalKTensorFlowApi::class)`.

## Tensor

### `Tensor` and `PhysicalTensor`

`Tensor` is now an interface with two kinds of implementations:

- `PhysicalTensor` owns its bytes and exposes them as `data`. The `Tensor(...)` factories return a `PhysicalTensor`.
- `TensorView` is a view of another tensor without its own memory: `reshape`, `flatten`, `transpose`, `permuted`, `squeeze` and `slice` return views.

`data` moved from `Tensor` to `PhysicalTensor`, and the concrete classes `FloatTensor`, `IntTensor`, `LongTensor` and `UByteTensor` are no longer public. Declare variables as `Tensor<Float>` or `PhysicalTensor<Float>` instead.

APIs that need the bytes accept a `PhysicalTensor`. Call `toPhysical()` on any other tensor: it returns the tensor itself when it is already physical and copies a view.

```kotlin
// 1.x
val bytes = tensor.data
interpreter.run(input, output)

// 2.0
val bytes = tensor.toPhysical().data
interpreter.run(input.toPhysical(), output)
val array = tensor.toPhysical().toArray<Array<FloatArray>>()
```

### Views share memory

In 1.x shape transformations returned new tensors. In 2.0 they return views over the same memory: writing through a view changes the original tensor, and the other way round. Call `toPhysical()` on a view when you need an independent copy.

Element access through a view goes through index translation on every call. For loops over every element of a large tensor, convert it with `toPhysical()` first; to fill a physical tensor, `setFlatUnboxed` writes without boxing the value.

### Stricter checks

These used to return wrong values silently and now fail:

- `Interpreter.run` with tensors checks their data type against the model and throws `TensorFlowException` on a mismatch, for example a `Tensor<Int>` passed to a `Float32` input.
- `get` and `set` by index check every axis: an index outside the shape throws `IndexOutOfBoundsException`, an index of the wrong rank throws `IllegalArgumentException`.
- `min`, `max`, `avg`, `argmin` and `argmax` of an empty tensor throw `NoSuchElementException`, like the standard library.
- `Tensor(dataType, shape, data)` requires `data` of exactly the size of the shape.

### Other changes

- `TensorShape` is a regular class instead of a value class. Source code is unaffected, but code compiled against 1.x must be recompiled.
- `toFlatIndex`, `toNestedIndex` and `strides` require `@OptIn(InternalKTensorFlowApi::class)`.
- `toString()` prints large tensors (more than 1000 elements) as a summary of their shape, type and edge values instead of every element.

## Pipeline

The pipeline module is experimental, so its API can still change. Notable changes:

- Outputs can be declared by signature name: `.output(name = "detection_boxes", ...)`.
- Postprocessing stages receive the model outputs in the order the outputs are declared.
- Declaring the same model output twice fails when the pipeline is built.
- `tuple(...)` creates the tuple a builder pipeline expects as its input.

## New in 2.0

These additions need no changes to existing code:

- `ktensorflow-vision`: cross-platform `Image`, tensorization, resizing, cropping, rotation, letterboxing, bounding boxes and NMS.
- `ktensorflow-coroutines`: `runSuspend`, `processFlow`, `processFlowDropping`, `SuspendInterpreter` and `InferenceDispatcher`.
- Model metadata: `getModelMeta()`, signature names and running with named outputs.
- `resizeInput`, which keeps the previous shape if the model cannot take the new one.
- Model loading from Compose resources packaged as Java resources or compressed assets, and from heap `ByteBuffer`s on Android.
