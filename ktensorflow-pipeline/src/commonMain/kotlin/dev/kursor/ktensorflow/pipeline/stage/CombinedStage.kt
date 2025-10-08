package dev.kursor.ktensorflow.pipeline.stage

import dev.kursor.ktensorflow.ExperimentalKTensorFlowApi
import dev.kursor.ktensorflow.pipeline.Tuple

@ExperimentalKTensorFlowApi
internal class CombinedStage<Input : Tuple, Output : Tuple>(
    val stages: List<Stage<Any?, Any?>>
) : Stage<Input, Output> {
    override fun run(input: Input): Output {
        return when (input) {
            is Tuple.Zero -> Tuple.Zero
            is Tuple.One<*> -> Tuple.One(
                stages[0].run(input.first)
            )

            is Tuple.Two<*, *> -> Tuple.Two(
                stages[0].run(input.first),
                stages[1].run(input.second)
            )

            is Tuple.Three<*, *, *> -> Tuple.Three(
                stages[0].run(input.first),
                stages[1].run(input.second),
                stages[2].run(input.third)
            )

            is Tuple.Four<*, *, *, *> -> Tuple.Four(
                stages[0].run(input.first),
                stages[1].run(input.second),
                stages[2].run(input.third),
                stages[3].run(input.fourth)
            )

            is Tuple.Five<*, *, *, *, *> -> Tuple.Five(
                stages[0].run(input.first),
                stages[1].run(input.second),
                stages[2].run(input.third),
                stages[3].run(input.fourth),
                stages[4].run(input.fifth)
            )

            is Tuple.Six<*, *, *, *, *, *> -> Tuple.Six(
                stages[0].run(input.first),
                stages[1].run(input.second),
                stages[2].run(input.third),
                stages[3].run(input.fourth),
                stages[4].run(input.fifth),
                stages[5].run(input.sixth)
            )

            is Tuple.Seven<*, *, *, *, *, *, *> -> Tuple.Seven(
                stages[0].run(input.first),
                stages[1].run(input.second),
                stages[2].run(input.third),
                stages[3].run(input.fourth),
                stages[4].run(input.fifth),
                stages[5].run(input.sixth),
                stages[6].run(input.seventh)
            )

            is Tuple.Eight<*, *, *, *, *, *, *, *> -> Tuple.Eight(
                stages[0].run(input.first),
                stages[1].run(input.second),
                stages[2].run(input.third),
                stages[3].run(input.fourth),
                stages[4].run(input.fifth),
                stages[5].run(input.sixth),
                stages[6].run(input.seventh),
                stages[7].run(input.eighth)
            )

            is Tuple.Nine<*, *, *, *, *, *, *, *, *> -> Tuple.Nine(
                stages[0].run(input.first),
                stages[1].run(input.second),
                stages[2].run(input.third),
                stages[3].run(input.fourth),
                stages[4].run(input.fifth),
                stages[5].run(input.sixth),
                stages[6].run(input.seventh),
                stages[7].run(input.eighth),
                stages[8].run(input.ninth)
            )

            is Tuple.Ten<*, *, *, *, *, *, *, *, *, *> -> Tuple.Ten(
                stages[0].run(input.first),
                stages[1].run(input.second),
                stages[2].run(input.third),
                stages[3].run(input.fourth),
                stages[4].run(input.fifth),
                stages[5].run(input.sixth),
                stages[6].run(input.seventh),
                stages[7].run(input.eighth),
                stages[8].run(input.ninth),
                stages[9].run(input.tenth)
            )
        } as Output
    }
}