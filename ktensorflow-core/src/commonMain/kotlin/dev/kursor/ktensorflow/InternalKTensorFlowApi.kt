package dev.kursor.ktensorflow

/**
 * Marks declarations that are public only because KTensorFlow modules share them with each other.
 *
 * They are not part of the stable API: they may change or disappear in any release, including a
 * minor one. Used as a subclassing requirement (`@SubclassOptInRequired`), it marks interfaces
 * that are meant to be used but not implemented outside the library, so that new members can be
 * added to them without breaking anyone.
 */
@RequiresOptIn(
    message = "This is an internal KTensorFlow API shared between its modules. " +
        "It may change or be removed in any release without notice.",
    level = RequiresOptIn.Level.ERROR
)
@Retention(AnnotationRetention.BINARY)
@Target(
    AnnotationTarget.CLASS,
    AnnotationTarget.FUNCTION,
    AnnotationTarget.PROPERTY,
    AnnotationTarget.CONSTRUCTOR,
    AnnotationTarget.TYPEALIAS
)
annotation class InternalKTensorFlowApi
