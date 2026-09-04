package org.anchor.contract

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.fail
import org.junit.jupiter.api.Test
import java.nio.file.Files
import java.nio.file.Path

/**
 * The tripwire ReplayCsvSchema own doc comment promises: if the committed
 * contracts/replay_csv/schema.json column list ever diverges from the
 * hardcoded Kotlin ReplayCsvSchema.COLUMNS, this test fails loudly instead
 * of the two silently drifting apart.
 *
 * Deliberately uses a narrow, hand-rolled extraction of "name": "..." in
 * declaration order rather than pulling in a JSON library dependency for
 * one synchronisation check -- schema.json own "columns" array is a simple,
 * known, stable shape (a list of objects each starting with a name field),
 * and this regex is scoped to exactly that, not a general JSON parser.
 */
class ReplayCsvSchemaSyncTest {

    @Test
    fun `ReplayCsvSchema COLUMNS matches the committed schema json exactly, in order`() {
        val schemaPath = findSchemaJson()
            ?: fail<Unit>("could not locate contracts/replay_csv/schema.json from the test working directory")
        val text = Files.readString(schemaPath as Path)

        val nameRegex = Regex("\"name\"\\s*:\\s*\"([a-zA-Z0-9_]+)\"")
        val namesInFile = nameRegex.findAll(text).map { it.groupValues[1] }.toList()

        assertEquals(
            namesInFile,
            ReplayCsvSchema.COLUMNS,
            "contracts/replay_csv/schema.json column list has changed and " +
                "ReplayCsvSchema.COLUMNS in Kotlin was not updated to match. " +
                "This is exactly the drift this test exists to catch -- " +
                "update ReplayCsvSchema.COLUMNS, do not edit this test.",
        )
    }

    private fun findSchemaJson(): Path? {
        val candidates = listOf(
            Path.of("../contracts/replay_csv/schema.json"),
            Path.of("contracts/replay_csv/schema.json"),
            Path.of("../../contracts/replay_csv/schema.json"),
        )
        return candidates.firstOrNull { Files.exists(it) }
    }
}
