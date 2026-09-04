package org.anchor.fusion

import org.anchor.math.Mat
import org.anchor.math.Vec
import org.anchor.math.Vec3

/**
 * Extension point for FR-27's GNSS position update, deliberately scoped
 * down this week (instruction: "do not fully implement GNSS ... unless
 * dependencies make it natural"). What IS real here: given a position
 * ALREADY expressed in local ENU metres, this is the simplest possible
 * MeasurementModel -- position is observed directly in the nav frame, no
 * rotation-dependent Jacobian at all (H = [I, 0, 0, 0, 0], simpler even
 * than NHC/ZUPT). What is genuinely NOT built, and should not be faked:
 * converting a raw GNSS lat/lon/alt fix into that local ENU frame needs a
 * chosen tangent-plane origin and a real geodetic projection -- separate,
 * real scope with its own correctness requirements, not something to
 * approximate inline here. Whoever wires live GNSS in supplies that
 * conversion upstream of this class; this class's job starts only once a
 * position is already in the filter's own nav frame.
 *
 * FR-27's chi-square gate is a SEPARATE concern from this class (see
 * ChiSquareGate) -- the orchestrator that eventually calls this should
 * run ChiSquareGate.test() on the computed innovation/S BEFORE calling
 * ErrorStateEkf.correct() with it, rejecting and logging a MODE_EVENT
 * rather than applying a fix that fails the gate. That orchestration
 * does not exist yet either (no EngineOrchestrator this week) -- both
 * pieces it would connect (this class, ChiSquareGate) are real and
 * tested independently now, so wiring them together later is composition,
 * not new algorithm design.
 */
class GnssPositionUpdate(
    private val positionEnu: Vec3,
    private val noiseVariance: Vec3,
) : MeasurementModel {
    override val name: String = "GnssPosition"
    override val dimension: Int = 3

    override fun predicted(state: NominalState): Vec =
        Vec.of(state.position.x, state.position.y, state.position.z)

    override fun actual(): Vec = Vec.of(positionEnu.x, positionEnu.y, positionEnu.z)

    override fun jacobian(state: NominalState): Mat {
        val h = Mat.zeros(3, ErrorStateLayout.DIM)
        for (i in 0 until 3) h[i, ErrorStateLayout.POS + i] = 1.0
        return h
    }

    override fun noise(): Mat = Mat.diagonal(doubleArrayOf(noiseVariance.x, noiseVariance.y, noiseVariance.z))
}
