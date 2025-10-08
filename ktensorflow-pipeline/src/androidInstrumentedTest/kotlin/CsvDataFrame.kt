import java.io.InputStream

class CsvDataFrame(
    inputStream: InputStream
) {
    private val reader = inputStream.bufferedReader()

    private val columns = reader
        .readLine()
        .split(",")
        .mapIndexed { index, string -> string to index}
        .toMap()

    init {
        println(columns)
    }

    fun <T> map(transformation: (CsvDataRow) -> T): List<T> {
        return reader.useLines { lines ->
            lines.map {
                transformation(CsvDataRow(it, columns))
            }
                .toList()
        }
    }
}

class CsvDataRow(
    line: String,
    private val columns: Map<String, Int>
) {
    private val data = line.split(",")
    operator fun get(key: String): String {
        return data[columns[key]!!]
    }
}