package org.anchor.contract

import org.json.JSONObject
import java.nio.file.Files
import java.nio.file.Path

/**
 * The single reader for contracts/model_io/model_manifest.json AND the
 * ONNX file own embedded metadata_props -- both converge here into one
 * representation, so nothing downstream ever touches raw JSON or a raw
 * metadata_props map directly, and no window size, feature name, or
 * normalisation constant is ever hand-copied as a Kotlin literal anywhere
 * else in this codebase (PRD-ANDROID-ENGINE.md Section7.2's explicit warning).
 *
 * Two construction paths, one internal shape:
 *  - fromManifestJson: used in local dev/tests where contracts/ is on
 *    disk. Reads the full committed model_manifest.json.
 *  - fromOnnxMetadata: used at real app runtime after a model has been
 *    loaded -- reads model.metadata_props (a flat String-to-String map),
 *    which is what VERSIONING.md own compatibility rule actually specifies
 *    ("the ONNX file own embedded contract_version") and what lets the app
 *    sanity-check a downloaded model with no second network call, since
 *    the contract travels inside the file itself. The embedded map does
 *    NOT carry feature_units or the outputs block (per the generator
 *    script own metadata dict) -- fromOnnxMetadata fills those from the
 *    FEATURE_ORDER/output-name conventions this contract has never changed
 *    (1.0.0), not by inventing values; a future contract_version bump that
 *    changes shape must extend the embedded metadata too, which is a
 *    VERSIONING.md MAJOR-bump discussion, not a Kotlin-side workaround.
 */
data class ModelManifest(
    val contractVersion: String,
    val windowSizeSamples: Int,
    val sampleRateHz: Int,
    val inputName: String,
    val inputShape: List<Int>,
    val featureOrder: List<String>,
    val normalizationMean: DoubleArray,
    val normalizationStd: DoubleArray,
    val outputMeanName: String,
    val outputLogVarianceName: String,
    val modelSha256: String?,
) {
    init {
        require(featureOrder.size == normalizationMean.size) {
            "featureOrder size (${featureOrder.size}) != normalizationMean size (${normalizationMean.size})"
        }
        require(featureOrder.size == normalizationStd.size) {
            "featureOrder size (${featureOrder.size}) != normalizationStd size (${normalizationStd.size})"
        }
    }

    /** (raw - mean) / std, per feature -- the ONE place this formula is
     *  applied. window is [timeSteps][features], features in featureOrder
     *  order. Returns a new array; does not mutate the input. */
    fun normalize(window: Array<DoubleArray>): Array<DoubleArray> = Array(window.size) { t ->
        DoubleArray(featureOrder.size) { f ->
            (window[t][f] - normalizationMean[f]) / normalizationStd[f]
        }
    }

    /**
     * VERSIONING.md compatibility rule, made executable: MAJOR must match
     * this.contractVersion exactly (a MAJOR bump is defined as a breaking
     * change), MINOR/PATCH of this manifest must be greater than or equal
     * to minSupported own MINOR/PATCH within that same MAJOR. Anything
     * else is a refusal, per the rule's own words: "never try anyway."
     */
    fun isCompatibleWith(minSupported: String): Boolean {
        val actual = SemVer.parse(contractVersion) ?: return false
        val min = SemVer.parse(minSupported) ?: return false
        if (actual.major != min.major) return false
        if (actual.minor != min.minor) return actual.minor > min.minor
        return actual.patch >= min.patch
    }

    companion object {
        // This project has exactly one model contract in play right now
        // (VERSIONING.md compatibility matrix: app 0.1.0 -> model_io
        // 1.0.0). This constant is the single named source every model-
        // load path must check against -- see the class doc and
        // PRD-ANDROID-ENGINE.md Section2.6 for why this line is described
        // there as "the single most important line of code for the whole
        // compatibility story."
        const val MIN_SUPPORTED_CONTRACT_VERSION = "1.0.0"

        fun fromManifestJson(path: Path): ModelManifest {
            val json = JSONObject(Files.readString(path))
            val normalization = json.getJSONObject("normalization")
            val outputs = json.getJSONObject("outputs")
            // Two output keys, contract-fixed names, order not assumed --
            // read by the exact names the contract specifies, never by
            // JSON object iteration order.
            return ModelManifest(
                contractVersion = json.getString("contract_version"),
                windowSizeSamples = json.getInt("window_size_samples"),
                sampleRateHz = json.getInt("sample_rate_hz"),
                inputName = json.getString("input_name"),
                inputShape = json.getJSONArray("input_shape").let { arr -> List(arr.length()) { arr.getInt(it) } },
                featureOrder = json.getJSONArray("feature_order").let { arr -> List(arr.length()) { arr.getString(it) } },
                normalizationMean = normalization.getJSONArray("mean").let { arr -> DoubleArray(arr.length()) { arr.getDouble(it) } },
                normalizationStd = normalization.getJSONArray("std").let { arr -> DoubleArray(arr.length()) { arr.getDouble(it) } },
                outputMeanName = outputs.keys().asSequence().first { it == "velocity_mean_mps" },
                outputLogVarianceName = outputs.keys().asSequence().first { it == "velocity_log_variance" },
                modelSha256 = json.optString("model_sha256", null),
            )
        }

        /**
         * metadataProps: the ONNX file own model.metadata_props, as key to
         * value strings, exactly as generate_stub_model.py writes them
         * (contract_version, window_size_samples, sample_rate_hz,
         * feature_order as a comma-joined string, output_mean_name,
         * output_logvar_name, norm_mean/norm_std as comma-joined strings).
         * Anything this map does not carry (input_name, input_shape,
         * modelSha256) is filled from the 1.0.0 contract own fixed
         * conventions -- input_name is always "imu_window" and input_shape
         * is always [1, windowSizeSamples, featureOrder.size] for this
         * contract version, per contracts/model_io/generate_stub_model.py.
         */
        fun fromOnnxMetadata(metadataProps: Map<String, String>): ModelManifest {
            fun require(key: String): String =
                metadataProps[key] ?: error("ONNX metadata_props missing required key: $key")

            val featureOrder = require("feature_order").split(",")
            val windowSize = require("window_size_samples").toInt()
            return ModelManifest(
                contractVersion = require("contract_version"),
                windowSizeSamples = windowSize,
                sampleRateHz = require("sample_rate_hz").toInt(),
                inputName = "imu_window",
                inputShape = listOf(1, windowSize, featureOrder.size),
                featureOrder = featureOrder,
                normalizationMean = require("norm_mean").split(",").map { it.toDouble() }.toDoubleArray(),
                normalizationStd = require("norm_std").split(",").map { it.toDouble() }.toDoubleArray(),
                outputMeanName = require("output_mean_name"),
                outputLogVarianceName = require("output_logvar_name"),
                modelSha256 = null, // computed separately from the file bytes, not embedded in itself
            )
        }
    }
}

internal data class SemVer(val major: Int, val minor: Int, val patch: Int) {
    companion object {
        fun parse(text: String): SemVer? {
            val parts = text.trim().split(".")
            if (parts.size != 3) return null
            val major = parts[0].toIntOrNull() ?: return null
            val minor = parts[1].toIntOrNull() ?: return null
            val patch = parts[2].toIntOrNull() ?: return null
            return SemVer(major, minor, patch)
        }
    }
}
