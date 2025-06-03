package dev.kursor.chess.engine.model

data class Square(
    val file: Int,
    val rank: Int
) {
    init {
        require(file in 0..7 && rank in 0..7) { "Invalid square: ($file, $rank)" }
    }

    override fun toString(): String = "${'a' + file}${rank + 1}"
}

val Square.isLight: Boolean
    get() = (file + rank) % 2 == 0