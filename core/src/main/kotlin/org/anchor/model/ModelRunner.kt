package org.anchor.model

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.anchor.contract.ModelManifest
import java.nio.file.Path

/**
 * API-SURFACE CAVEAT, stated plainly rather than hidden: the exact
 * ai.onnxruntime method names below (session.metadata / .customMetadata,
 * OnnxTensor.createTensor's array-inferring overload, OnnxValue.value's
 * cast shape) are written from training knowledge of the ONNX Runtime
 * Java API, not verified against onnxruntime 1.29.0's actual API docs --
 * this dev environment has no JDK/Gradle to compile against, so I cannot
 * check them live. Confirm these exact calls on the FIRST real compile;
 * treat this file as "correct by design, unverified by the compiler" per
 * the Week-1 report, not as tested code.
 *
 * FR-07 provenance note: the inference path below reads ONLY the aligned
 * IMU window and the manifest -- no GNSS, no wheel-speed, nothing outside
 * org.anchor.contract.ModelManifest's declared feature_order ever reaches
 * the input tensor. A future provenance-assertion test (mentioned in
 * PRD-ANDROID-ENGINE.md FR-07) should build on this file's single input
 * construction site, not duplicate it.
 */
class ModelContractRefusedException(message: String) : Exception(message)

class ModelRunner private constructor(
    private val session: OrtSession,
    private val environment: OrtEnvironment,
    val manifest: ModelManifest,
) : AutoCloseable {

    /**
     * Per FR-11: mean is m/s (VERSIONING.md "Resolved cross-track
     * questions", confirmed 2026-09-03 -- NOT displacement, see this
     * session own report). variance = exp(velocity_log_variance), applied
     * in exactly this one place -- PRD-ANDROID-ENGINE.md Section2.2/Section7.2's
     * explicit instruction against scattering that exp() call.
     */
    data class InferenceResult(val velocityMeanMps: Double, val velocityVariance: Double)

    /**
     * alignedWindow: [windowSizeSamples][featureOrder.size], already
     * vehicle-frame, gravity-removed (org.anchor.alignment's job, strictly
     * upstream of this call) and NOT yet normalised -- normalisation
     * happens inside, reading manifest.normalizationMean/Std, matching
     * model_manifest.json's own normalization.applied_by = "caller,
     * before inference".
     */
    fun infer(alignedWindow: Array<DoubleArray>): InferenceResult {
        require(alignedWindow.size == manifest.windowSizeSamples) {
            "window has ${alignedWindow.size} timesteps, manifest requires ${manifest.windowSizeSamples}"
        }
        require(alignedWindow.all { it.size == manifest.featureOrder.size }) {
            "window timestep feature count does not match manifest featureOrder size ${manifest.featureOrder.size}"
        }

        val normalized = manifest.normalize(alignedWindow)
        // [1][windowSize][features] float32, matching input_shape exactly --
        // built from manifest fields, never a hardcoded [1,20,6] literal.
        val inputArray = Array(1) {
            Array(manifest.windowSizeSamples) { t ->
                FloatArray(manifest.featureOrder.size) { f -> normalized[t][f].toFloat() }
            }
        }

        OnnxTensor.createTensor(environment, inputArray).use { inputTensor ->
            session.run(mapOf(manifest.inputName to inputTensor)).use { results ->
                val meanValue = extractScalar(results, manifest.outputMeanName)
                val logVarValue = extractScalar(results, manifest.outputLogVarianceName)
                return InferenceResult(
                    velocityMeanMps = meanValue,
                    velocityVariance = kotlin.math.exp(logVarValue),
                )
            }
        }
    }

    @Suppress("UNCHECKED_CAST")
    private fun extractScalar(results: OrtSession.Result, outputName: String): Double {
        val value = results.get(outputName)
            .orElseThrow { IllegalStateException("model produced no output named $outputName") }
        val raw = (value as OnnxTensor).value
        val array2d = raw as Array<FloatArray> // shape [1,1] per the contract
        return array2d[0][0].toDouble()
    }

    override fun close() {
        session.close()
    }

    companion object {
        /**
         * The single load path every model-load code path must go
         * through -- first launch, OTA update, replay harness model swap
         * (PRD-ANDROID-ENGINE.md Section2.6). Reads the manifest from the
         * loaded model own embedded metadata_props (VERSIONING.md
         * compatibility rule's authoritative source, no second file or
         * network call needed), checks it against
         * ModelManifest.MIN_SUPPORTED_CONTRACT_VERSION, and REFUSES --
         * throws, does not return a usable ModelRunner -- on any
         * incompatibility. There is deliberately no lower-level
         * constructor callers could reach to bypass this.
         */
        fun load(modelPath: Path): ModelRunner {
            val env = OrtEnvironment.getEnvironment()
            val session = env.createSession(modelPath.toString(), OrtSession.SessionOptions())

            val customMetadata: Map<String, String> = session.metadata.customMetadata
            val manifest = ModelManifest.fromOnnxMetadata(customMetadata)

            if (!manifest.isCompatibleWith(ModelManifest.MIN_SUPPORTED_CONTRACT_VERSION)) {
                session.close()
                throw ModelContractRefusedException(
                    "model at $modelPath has contract_version=${manifest.contractVersion}, " +
                        "not compatible with MIN_SUPPORTED_CONTRACT_VERSION=" +
                        "${ModelManifest.MIN_SUPPORTED_CONTRACT_VERSION}. Refusing to load per " +
                        "VERSIONING.md own compatibility-refusal rule -- never try anyway. " +
                        "Caller must fall back per FR-24 (degrade to NHC-only, degraded pill, log).",
                )
            }

            return ModelRunner(session, env, manifest)
        }
    }
}
