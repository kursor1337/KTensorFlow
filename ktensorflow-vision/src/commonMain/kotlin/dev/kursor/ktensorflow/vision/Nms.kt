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
 * The result is sorted by descending score, with ties kept in input order. [scoreSelector] is
 * called once per item and [boxSelector] once per item that passes [scoreThreshold].
 *
 * @param T The type of elements in the collection.
 * @param C The type of class labels.
 * @param iouThreshold The threshold for the Intersection over Union (IoU) metric. Boxes with an IoU
 * higher than this value relative to a higher-scoring box will be suppressed. Defaults to 0.45.
 * @param scoreThreshold The minimum confidence score required to keep a box. Defaults to 0.25.
 * @param scoreSelector A function to extract the confidence score from an element.
 * @param boxSelector A function to extract the bounding box from an element.
 * @param classSelector A function to extract the class label, or null to suppress across all classes.
 */
fun <T, C> Iterable<T>.nms(
    iouThreshold: Float = 0.45f,
    scoreThreshold: Float = 0.25f,
    scoreSelector: (T) -> Float,
    boxSelector: (T) -> Rect,
    classSelector: ((T) -> C)? = null
): List<T> {
    // Score и бокс каждого элемента считаются ровно один раз: сортировка звала scoreSelector
    // на каждое сравнение, а внутренний цикл - boxSelector для каждой пары, то есть O(n^2) раз
    val candidates = mapNotNull { item ->
        val score = scoreSelector(item)
        if (score >= scoreThreshold) Candidate(item, score, boxSelector(item)) else null
    }
    if (candidates.isEmpty()) return emptyList()

    val kept = if (classSelector != null) {
        candidates
            .groupBy { classSelector(it.item) }
            .flatMap { (_, group) -> group.nmsSingleClass(iouThreshold) }
            // Без общей сортировки результат шёл группами по классам, а не по уверенности
            .sortedByDescending { it.score }
    } else {
        candidates.nmsSingleClass(iouThreshold)
    }

    return kept.map { it.item }
}

private class Candidate<T>(val item: T, val score: Float, val box: Rect)

/** Возвращает оставшиеся кандидаты по убыванию score; при равных score порядок входа сохраняется. */
private fun <T> List<Candidate<T>>.nmsSingleClass(iouThreshold: Float): List<Candidate<T>> {
    val sorted = sortedByDescending { it.score }
    val active = BooleanArray(sorted.size) { true }
    val results = mutableListOf<Candidate<T>>()

    for (i in sorted.indices) {
        if (!active[i]) continue

        val current = sorted[i]
        results.add(current)

        for (j in i + 1 until sorted.size) {
            if (active[j] && current.box.intersectionOverUnion(sorted[j].box) > iouThreshold) {
                active[j] = false
            }
        }
    }

    return results
}
