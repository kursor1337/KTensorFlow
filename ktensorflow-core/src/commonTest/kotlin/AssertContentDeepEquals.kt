fun <T> assertContentDeepEquals(
    expected: Array<T>,
    actual: Array<T>
) {
    if (expected.size != actual.size) {
        throw AssertionError("Arrays are not the same size: ${expected.size} != ${actual.size}")
    }
    if (!expected.contentDeepEquals(actual)) {
        throw AssertionError(
            "Arrays are not equal: " +
                    expected.contentDeepToString() +
                    " != " +
                    actual.contentDeepToString()
        )
    }
}
