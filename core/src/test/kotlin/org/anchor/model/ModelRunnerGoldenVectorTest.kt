package org.anchor.model

import org.json.JSONObject
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Assertions.fail
import org.junit.jupiter.api.DynamicTest
import org.junit.jupiter.api.TestFactory
import org.junit.jupiter.api.condition.EnabledIfSystemProperty
import java.nio.file.Files
import java.nio.file.Path

/**
 * THIS is the desktop half of the anti-drift mechanism PRD-ANDROID-ENGINE.md
 * Section2.3/Section7.7 assigns to this track -- runs on the DESKTOP onnxruntime
 * artifact (TOOLCHAIN.md "ONNX Runtime -- desktop" row), so it is a plain
 * JUnit test needing no emulator or connected device. It is NOT the
 * instrumented android-contract-check job in contracts-ci.yml (that one
 * needs onnxruntime-android on a real/emulated device and is still to
 * build) -- but it exercises the exact same ModelRunner class, the exact
 * same committed anchor_net_stub.onnx, and the exact same committed
 * golden_vectors/*.json, so it is real, independent evidence the Kotlin
 * port of the contract-reading and inference path is correct, before an
 * Android device is even available to test on.
 *
 * NOTE: written, not run -- see ReplayCsvParserTest for why. Gated behind
 * a system property so a plain `./gradlew test` does not hard-fail on a
 * machine where contracts/ is not checked out (a published :core artifact
 * consumed standalone, for instance) -- run with
 * `./gradlew :core:test -Danchor.goldenVectors=true`.
 */
@EnabledIfSystemProperty(named = "anchor.goldenVectors", matches = "true")
class ModelRunnerGoldenVectorTest {

    @TestFactory
    fun `every committed golden vector reproduces within its own tolerance_abs`(): List<DynamicTest> {
        val modelPath = findRepoRelative("contracts/model_io/anchor_net_stub.onnx")
            ?: fail<Unit>("could not locate contracts/model_io/anchor_net_stub.onnx from the test working directory")
        val vectorsDir = findRepoRelative("contracts/model_io/golden_vectors")
            ?: fail<Unit>("could not locate contracts/model_io/golden_vectors from the test working directory")

        val vectorFiles = Files.list(vectorsDir as Path).use { it.filter { p -> p.toString().endsWith(".json") }.sorted().toList() }
        assertTrue(vectorFiles.isNotEmpty(), "no golden vector files found in $vectorsDir")

        return vectorFiles.map { vectorFile ->
            DynamicTest.dynamicTest(vectorFile.fileName.toString()) {
                ModelRunner.load(modelPath as Path).use { runner ->
                    val json = JSONObject(Files.readString(vectorFile))
                    val inputArray = json.getJSONArray("input")
                    val window = Array(inputArray.length()) { t ->
                        val row = inputArray.getJSONArray(t)
                        DoubleArray(row.length()) { f -> row.getDouble(f) }
                    }
                    val expected = json.getJSONObject("expected_output")
                    val tolerance = json.getDouble("tolerance_abs")

                    val result = runner.infer(window)

                    // NOTE: expected_output.velocity_log_variance is the raw log-variance the
                    // model emits, compared here against the pre-exp() value -- ModelRunner
                    // itself only ever returns the post-exp() variance (its documented
                    // contract), so this test recomputes ln(result.velocityVariance) to
                    // compare like-for-like against the committed golden value, rather than
                    // exposing a raw/unconverted accessor on ModelRunner just for this test.
                    assertEquals(
                        expected.getDouble("velocity_mean_mps"),
                        result.velocityMeanMps,
                        tolerance,
                        "${vectorFile.fileName}: velocity_mean_mps drifted",
                    )
                    assertEquals(
                        expected.getDouble("velocity_log_variance"),
                        kotlin.math.ln(result.velocityVariance),
                        tolerance,
                        "${vectorFile.fileName}: velocity_log_variance drifted",
                    )
                }
            }
        }
    }

    private fun findRepoRelative(relative: String): Path? {
        val candidates = listOf(
            Path.of("../$relative"),
            Path.of(relative),
            Path.of("../../$relative"),
        )
        return candidates.firstOrNull { Files.exists(it) }
    }
}
