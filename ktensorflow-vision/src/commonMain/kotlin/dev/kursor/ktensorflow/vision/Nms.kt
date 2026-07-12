package dev.kursor.ktensorflow.vision

/**
 * Performs Non-Maximum Suppression (NMS) on a collection of objects to filter out redundant,
 * overlapping bounding boxes.
 *
 * This function first filters out items below the [scoreThreshold]. It then iteratively selects
 * the highest-scoring boxes and removes any remaining boxes that have an Intersection over Union (IoU)
 * greater than the [iouThreshold] with the selected box.
 *
 * If a [classSelector] is provided, the NMS process is applied independently to each class
 * (Multi-class NMS). Otherwise, it is applied globally across all items.
 *
 * @param T The type of elements in the collection.
 * @param iouThreshold The threshold for the Intersection over Union (IoU) metric. Boxes with an IoU
 * higher than this value relative to a higher-scoring box will be suppressed. Defaults to 0.45.
 * @param scoreThreshold The minimum confidence score required to keep a box. Defaults to 0.25.
 * @param scoreSelector A function to extract the confidence score from an element.
 */
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