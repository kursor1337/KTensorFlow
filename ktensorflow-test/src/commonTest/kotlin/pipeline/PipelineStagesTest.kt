package pipeline

import dev.kursor.ktensorflow.ExperimentalKTensorFlowApi
import dev.kursor.ktensorflow.pipeline.Pipeline
import dev.kursor.ktensorflow.pipeline.Tuple
import dev.kursor.ktensorflow.pipeline.linear
import dev.kursor.ktensorflow.pipeline.stage.ArgmaxStage
import dev.kursor.ktensorflow.pipeline.stage.ClassificationStage
import dev.kursor.ktensorflow.pipeline.stage.FilterStage
import dev.kursor.ktensorflow.pipeline.stage.IndexedFilterStage
import dev.kursor.ktensorflow.pipeline.stage.Stage
import dev.kursor.ktensorflow.pipeline.stage.ValuedArgmaxStage
import dev.kursor.ktensorflow.pipeline.stage.argmax
import dev.kursor.ktensorflow.pipeline.stage.classify
import dev.kursor.ktensorflow.pipeline.stage.filter
import dev.kursor.ktensorflow.pipeline.stage.filterIndexed
import dev.kursor.ktensorflow.pipeline.stage.filterNot
import dev.kursor.ktensorflow.pipeline.stage.then
import dev.kursor.ktensorflow.pipeline.stage.valuedArgmax
import dev.kursor.ktensorflow.pipeline.tuple
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertSame

/**
 * Постобработка пайплайна - обычный Kotlin-код без интерпретатора, но именно она превращает
 * сырой выход модели в доменную модель, поэтому здесь проверяются и нормальные случаи, и
 * границы: пустой вход, ничего не прошедший фильтр, одинаковые максимумы, выход за диапазон.
 */
@OptIn(ExperimentalKTensorFlowApi::class)
class PipelineStagesTest {

    // --- фильтрация ---

    @Test
    fun filterStageKeepsOnlyMatchingElements() {
        val stage = FilterStage<Int> { it % 2 == 0 }

        assertEquals(listOf(2, 4, 6), stage.run(listOf(1, 2, 3, 4, 5, 6)))
    }

    @Test
    fun filterStageReturnsAnEmptyListWhenNothingMatches() {
        val stage = FilterStage<Int> { it > 100 }

        assertEquals(emptyList(), stage.run(listOf(1, 2, 3)))
    }

    @Test
    fun filterStageOnAnEmptyInputReturnsAnEmptyList() {
        val stage = FilterStage<Int> { true }

        assertEquals(emptyList(), stage.run(emptyList()))
    }

    @Test
    fun filterOperatorFiltersTheOutputOfThePrecedingStage() {
        val stage = Stage<Int, List<Int>> { count -> (1..count).toList() }
            .filter { it % 3 == 0 }

        assertEquals(listOf(3, 6, 9), stage.run(10))
    }

    @Test
    fun filterNotIsTheExactComplementOfFilter() {
        val source = Stage<Unit, List<Int>> { (1..6).toList() }
        val predicate: (Int) -> Boolean = { it % 2 == 0 }

        val kept = source.filter(predicate).run(Unit)
        val dropped = source.filterNot(predicate).run(Unit)

        assertEquals(listOf(2, 4, 6), kept)
        assertEquals(listOf(1, 3, 5), dropped)
        assertEquals(listOf(1, 2, 3, 4, 5, 6), (kept + dropped).sorted())
    }

    @Test
    fun indexedFilterStageSeesTheOriginalPositionOfEachElement() {
        // Позиция должна быть позицией во ВХОДНОМ списке, а не в результате
        val stage = IndexedFilterStage<String> { index, _ -> index % 2 == 0 }

        assertEquals(listOf("a", "c", "e"), stage.run(listOf("a", "b", "c", "d", "e")))
    }

    @Test
    fun filterIndexedOperatorFiltersByPositionAndValue() {
        val stage = Stage<Unit, List<Int>> { listOf(10, 20, 30, 40) }
            .filterIndexed { index, value -> index > 0 && value < 40 }

        assertEquals(listOf(20, 30), stage.run(Unit))
    }

    // --- argmax ---

    @Test
    fun argmaxStageReturnsTheIndexOfTheLargestValue() {
        val stage = ArgmaxStage()

        assertEquals(2, stage.run(floatArrayOf(0.1f, 0.2f, 0.9f, 0.3f)))
    }

    @Test
    fun argmaxStageReturnsTheFirstIndexWhenValuesAreTied() {
        val stage = ArgmaxStage()

        assertEquals(1, stage.run(floatArrayOf(0.1f, 0.9f, 0.9f, 0.4f)))
    }

    @Test
    fun argmaxStageHandlesASingleElementAndNegativeValues() {
        val stage = ArgmaxStage()

        assertEquals(0, stage.run(floatArrayOf(-5f)))
        assertEquals(1, stage.run(floatArrayOf(-5f, -1f, -3f)))
    }

    @Test
    fun argmaxStageFailsOnAnEmptyOutput() {
        // Пустой выход модели - ошибка конфигурации пайплайна, и она должна быть громкой,
        // а не молча превращаться в индекс 0
        assertFailsWith<NoSuchElementException> { ArgmaxStage().run(floatArrayOf()) }
    }

    @Test
    fun argmaxOperatorChainsAfterThePrecedingStage() {
        val stage = Stage<Int, FloatArray> { size -> FloatArray(size) { it.toFloat() } }
            .argmax()

        assertEquals(4, stage.run(5))
    }

    @Test
    fun valuedArgmaxStageReturnsBothIndexAndValue() {
        val stage = ValuedArgmaxStage()

        assertEquals(2 to 0.9f, stage.run(floatArrayOf(0.1f, 0.2f, 0.9f, 0.3f)))
    }

    @Test
    fun valuedArgmaxAgreesWithArgmaxOnTheSameInput() {
        val scores = floatArrayOf(0.05f, 0.7f, 0.2f, 0.05f)

        val index = ArgmaxStage().run(scores)
        val (valuedIndex, value) = ValuedArgmaxStage().run(scores)

        assertEquals(index, valuedIndex)
        assertEquals(scores[index], value)
    }

    @Test
    fun valuedArgmaxOperatorChainsAfterThePrecedingStage() {
        val stage = Stage<Unit, FloatArray> { floatArrayOf(0.2f, 0.8f) }.valuedArgmax()

        assertEquals(1 to 0.8f, stage.run(Unit))
    }

    // --- классификация ---

    @Test
    fun classificationStageMapsAnIndexToItsClass() {
        val stage = ClassificationStage(listOf("cat", "dog", "bird"))

        assertEquals("dog", stage.run(1))
    }

    @Test
    fun classificationStageFailsWhenTheIndexIsOutsideTheClassList() {
        // Модель с 10 выходами и список из 3 классов - ошибка конфигурации, она не должна
        // возвращать произвольный класс
        val stage = ClassificationStage(listOf("cat", "dog", "bird"))

        assertFailsWith<IndexOutOfBoundsException> { stage.run(3) }
    }

    @Test
    fun argmaxAndClassifyFormTheUsualClassificationTail() {
        val classes = listOf("zero", "one", "two")
        val stage = Stage<Unit, FloatArray> { floatArrayOf(0.1f, 0.2f, 0.7f) }
            .argmax()
            .classify(classes)

        assertEquals("two", stage.run(Unit))
    }

    // --- составление стадий ---

    @Test
    fun thenAppliesStagesInOrderNotInReverse() {
        val stage = Stage<Int, Int> { it + 1 }
            .then(Stage<Int, Int> { it * 10 })

        // (1 + 1) * 10 = 20, а не 1 + 1 * 10 = 11
        assertEquals(20, stage.run(1))
    }

    @Test
    fun thenPropagatesAFailureFromTheInnerStageWithoutRunningTheOuterOne() {
        var outerRan = false
        val stage = Stage<Int, Int> { error("inner failed") }
            .then(Stage<Int, Int> { outerRan = true; it })

        assertFailsWith<IllegalStateException> { stage.run(1) }
        assertEquals(false, outerRan)
    }

    @Test
    fun noopStageReturnsTheVerySameInstance() {
        val stage = Stage<List<String>>()
        val input = listOf("a", "b")

        assertSame(input, stage.run(input))
    }

    @Test
    fun linearPipelineIsANoopUntilStagesAreChained() {
        val pipeline = Pipeline(Pipeline.linear<Int>())

        assertEquals(42, pipeline.run(42))
    }

    // --- Tuple ---

    @Test
    fun tupleFactoriesBuildTheArityTheyAreGiven() {
        assertEquals(Tuple.Zero, tuple())
        assertEquals(Tuple.One(1), tuple(1))
        assertEquals(Tuple.Two(1, 2), tuple(1, 2))
        assertEquals(Tuple.Three(1, 2, 3), tuple(1, 2, 3))
        assertEquals(Tuple.Four(1, 2, 3, 4), tuple(1, 2, 3, 4))
        assertEquals(Tuple.Five(1, 2, 3, 4, 5), tuple(1, 2, 3, 4, 5))
        assertEquals(Tuple.Six(1, 2, 3, 4, 5, 6), tuple(1, 2, 3, 4, 5, 6))
        assertEquals(Tuple.Seven(1, 2, 3, 4, 5, 6, 7), tuple(1, 2, 3, 4, 5, 6, 7))
        assertEquals(Tuple.Eight(1, 2, 3, 4, 5, 6, 7, 8), tuple(1, 2, 3, 4, 5, 6, 7, 8))
        assertEquals(Tuple.Nine(1, 2, 3, 4, 5, 6, 7, 8, 9), tuple(1, 2, 3, 4, 5, 6, 7, 8, 9))
        assertEquals(
            Tuple.Ten(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
            tuple(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        )
    }

    @Test
    fun tupleComponentsAreExposedInTheDeclaredOrder() {
        // Порядок полей - это порядок выходов модели, перепутанные позиции незаметны,
        // пока каждое поле не проверено отдельно
        val ten = tuple(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

        assertContentEquals(
            (0..9).toList(),
            listOf(
                ten.first,
                ten.second,
                ten.third,
                ten.fourth,
                ten.fifth,
                ten.sixth,
                ten.seventh,
                ten.eighth,
                ten.ninth,
                ten.tenth
            )
        )
    }

    @Test
    fun tuplesOfDifferentArityAreNotEqual() {
        assertEquals(false, tuple(1).equals(tuple(1, 1)))
        assertEquals(false, Tuple.Zero.equals(tuple(1)))
    }
}
