package dev.kursor.ktensorflow.media

fun <T> Iterable<T>.nms(
    iouThreshold: Float = 0.45f,
    scoreThreshold: Float = 0.25f,
    scoreSelector: (T) -> Float,
    boxSelector: (T) -> Rect,
    classSelector: ((T) -> Int)? = null
): List<T> {
    val validItems = this.filter { scoreSelector(it) >= scoreThreshold }
    if (validItems.isEmpty()) return emptyList()

    // if classSelector was passed, group and filter each class separately
    if (classSelector != null) {
        return validItems
            .groupBy(classSelector)
            .flatMap { (_, items) -> items.nmsSingleClass(iouThreshold, scoreSelector, boxSelector) }
    }

    return validItems.nmsSingleClass(iouThreshold, scoreSelector, boxSelector)
}

private fun <T> List<T>.nmsSingleClass(
    iouThreshold: Float,
    scoreSelector: (T) -> Float,
    boxSelector: (T) -> Rect
): List<T> {
    // sort by descending score
    val sorted = this.sortedByDescending(scoreSelector)
    val active = BooleanArray(sorted.size) { true }
    val results = mutableListOf<T>()

    for (i in sorted.indices) {
        if (!active[i]) continue
        
        val current = sorted[i]
        val currentBox = boxSelector(current)
        results.add(current)

        for (j in i + 1 until sorted.size) {
            if (active[j]) {
                val otherBox = boxSelector(sorted[j])
                if (currentBox.intersectionOverUnion(otherBox) > iouThreshold) {
                    active[j] = false
                }
            }
        }
    }
    
    return results
}