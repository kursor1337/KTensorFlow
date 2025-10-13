class CsvDataFrame(
    private val text: String
) {

    private val columns = text
        .lines()
        .first()
        .split(",")
        .mapIndexed { index, string -> string to index }
        .toMap()

    fun <T> map(transformation: (CsvDataRow) -> T): List<T> {
        return text
            .trim()
            .lines()
            .drop(1)
            .map { transformation(CsvDataRow( it, columns)) }
            .toList()
    }
}

class CsvDataRow(
    line: String,
    private val columns: Map<String, Int>
) {
    val data = line.split(",")
    operator fun get(key: String): String {
        return data[columns[key]!!]
    }
}