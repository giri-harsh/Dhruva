package org.anchor.math

/**
 * General R x C matrix, row-major flat DoubleArray storage. The ESKF own
 * 15x15 covariance, its Jacobians (as small as 1x15 for VelocityUpdate, as
 * large as 15x15 for the state-transition matrix), and small measurement-
 * space matrices (2x2 for NHC, 3x3 for ZUPT) all go through this type.
 *
 * Allocating, readable operators throughout -- correctness and testability
 * first, per this week own instruction; not the zero-allocation hot-path
 * variant the PRD schedules for Week 3 (Section5.3).
 */
class Mat(val rows: Int, val cols: Int, val data: DoubleArray) {
    init {
        require(data.size == rows * cols) {
            "Mat data size ${data.size} does not match rows*cols = ${rows * cols}"
        }
    }

    operator fun get(r: Int, c: Int): Double {
        require(r in 0 until rows && c in 0 until cols) { "index ($r,$c) out of bounds for ${rows}x$cols" }
        return data[r * cols + c]
    }

    operator fun set(r: Int, c: Int, v: Double) {
        require(r in 0 until rows && c in 0 until cols) { "index ($r,$c) out of bounds for ${rows}x$cols" }
        data[r * cols + c] = v
    }

    operator fun plus(other: Mat): Mat {
        require(rows == other.rows && cols == other.cols) { "Mat shape mismatch: ${rows}x$cols vs ${other.rows}x${other.cols}" }
        return Mat(rows, cols, DoubleArray(data.size) { data[it] + other.data[it] })
    }

    operator fun minus(other: Mat): Mat {
        require(rows == other.rows && cols == other.cols) { "Mat shape mismatch: ${rows}x$cols vs ${other.rows}x${other.cols}" }
        return Mat(rows, cols, DoubleArray(data.size) { data[it] - other.data[it] })
    }

    operator fun times(scalar: Double): Mat = Mat(rows, cols, DoubleArray(data.size) { data[it] * scalar })

    operator fun times(other: Mat): Mat {
        require(cols == other.rows) { "Mat multiply shape mismatch: ${rows}x$cols times ${other.rows}x${other.cols}" }
        val result = DoubleArray(rows * other.cols)
        for (i in 0 until rows) {
            for (k in 0 until cols) {
                val aik = data[i * cols + k]
                if (aik == 0.0) continue
                val otherRowOffset = k * other.cols
                val resultRowOffset = i * other.cols
                for (j in 0 until other.cols) {
                    result[resultRowOffset + j] += aik * other.data[otherRowOffset + j]
                }
            }
        }
        return Mat(rows, other.cols, result)
    }

    operator fun times(v: Vec): Vec {
        require(cols == v.size) { "Mat*Vec shape mismatch: ${rows}x$cols times ${v.size}" }
        val result = DoubleArray(rows)
        for (i in 0 until rows) {
            var s = 0.0
            val rowOffset = i * cols
            for (j in 0 until cols) s += data[rowOffset + j] * v[j]
            result[i] = s
        }
        return Vec(result)
    }

    fun transpose(): Mat {
        val result = DoubleArray(rows * cols)
        for (i in 0 until rows) {
            for (j in 0 until cols) {
                result[j * rows + i] = data[i * cols + j]
            }
        }
        return Mat(cols, rows, result)
    }

    /** In-place symmetrization: (M + M^T) / 2. Used after Joseph-form
     *  covariance updates, where floating-point rounding can otherwise
     *  accumulate a tiny asymmetry over many propagate/update cycles. */
    fun symmetrized(): Mat {
        require(rows == cols) { "symmetrized() requires a square matrix, got ${rows}x$cols" }
        val result = DoubleArray(data.size)
        for (i in 0 until rows) {
            for (j in 0 until cols) {
                val avg = (data[i * cols + j] + data[j * cols + i]) / 2.0
                result[i * cols + j] = avg
            }
        }
        return Mat(rows, cols, result)
    }

    /** Reads the [r0,c0)..[r0+h,c0+w) sub-block as a new Mat. */
    fun block(r0: Int, c0: Int, h: Int, w: Int): Mat {
        require(r0 + h <= rows && c0 + w <= cols) { "block ($r0,$c0,$h,$w) out of bounds for ${rows}x$cols" }
        val result = DoubleArray(h * w)
        for (i in 0 until h) {
            for (j in 0 until w) {
                result[i * w + j] = data[(r0 + i) * cols + (c0 + j)]
            }
        }
        return Mat(h, w, result)
    }

    /** Writes [block] into this matrix starting at (r0, c0), in place. */
    fun setBlock(r0: Int, c0: Int, block: Mat) {
        require(r0 + block.rows <= rows && c0 + block.cols <= cols) {
            "setBlock at ($r0,$c0) with shape ${block.rows}x${block.cols} out of bounds for ${rows}x$cols"
        }
        for (i in 0 until block.rows) {
            for (j in 0 until block.cols) {
                data[(r0 + i) * cols + (c0 + j)] = block.data[i * block.cols + j]
            }
        }
    }

    /** Writes a Mat3 rotation/skew block into this matrix at (r0, c0). */
    fun setBlock3(r0: Int, c0: Int, m3: Mat3) {
        setBlock(
            r0, c0,
            Mat(3, 3, doubleArrayOf(m3.m00, m3.m01, m3.m02, m3.m10, m3.m11, m3.m12, m3.m20, m3.m21, m3.m22)),
        )
    }

    /**
     * Matrix inverse via Gauss-Jordan elimination with partial pivoting.
     * Throws IllegalStateException on a singular (or near-singular, by a
     * fixed pivot-magnitude floor) matrix rather than returning garbage --
     * a silently-wrong inverse is exactly the kind of bug that would show
     * up as a plausible-looking but wrong Kalman gain, not a crash.
     */
    fun inverse(): Mat {
        require(rows == cols) { "inverse() requires a square matrix, got ${rows}x$cols" }
        val n = rows
        // Augmented [A | I], worked on as a scratch copy -- allocating here
        // is fine, this runs on small (<=15x15) measurement-space matrices,
        // not in the propagate() hot path.
        val aug = Array(n) { i -> DoubleArray(2 * n).also { row ->
            for (j in 0 until n) row[j] = data[i * n + j]
            row[n + i] = 1.0
        } }

        for (col in 0 until n) {
            var pivotRow = col
            var pivotVal = kotlin.math.abs(aug[col][col])
            for (r in col + 1 until n) {
                val v = kotlin.math.abs(aug[r][col])
                if (v > pivotVal) { pivotVal = v; pivotRow = r }
            }
            check(pivotVal > 1e-12) { "Mat.inverse(): matrix is singular or near-singular (pivot=$pivotVal at column $col)" }
            if (pivotRow != col) {
                val tmp = aug[col]; aug[col] = aug[pivotRow]; aug[pivotRow] = tmp
            }
            val pivot = aug[col][col]
            for (j in 0 until 2 * n) aug[col][j] /= pivot
            for (r in 0 until n) {
                if (r == col) continue
                val factor = aug[r][col]
                if (factor == 0.0) continue
                for (j in 0 until 2 * n) aug[r][j] -= factor * aug[col][j]
            }
        }

        val result = DoubleArray(n * n)
        for (i in 0 until n) {
            for (j in 0 until n) result[i * n + j] = aug[i][n + j]
        }
        return Mat(n, n, result)
    }

    fun copy(): Mat = Mat(rows, cols, data.copyOf())

    override fun toString(): String = buildString {
        for (i in 0 until rows) {
            append((0 until cols).joinToString(prefix = "[", postfix = "]", separator = ", ") { j -> "%.4g".format(data[i * cols + j]) })
            if (i < rows - 1) append("\n")
        }
    }

    companion object {
        fun zeros(rows: Int, cols: Int) = Mat(rows, cols, DoubleArray(rows * cols))

        fun identity(n: Int): Mat {
            val d = DoubleArray(n * n)
            for (i in 0 until n) d[i * n + i] = 1.0
            return Mat(n, n, d)
        }

        fun diagonal(values: DoubleArray): Mat {
            val n = values.size
            val d = DoubleArray(n * n)
            for (i in 0 until n) d[i * n + i] = values[i]
            return Mat(n, n, d)
        }

        /** Skew-symmetric cross-product matrix [v]_x such that [v]_x * w == v cross w. */
        fun skew(v: Vec3): Mat3 = Mat3(
            0.0, -v.z, v.y,
            v.z, 0.0, -v.x,
            -v.y, v.x, 0.0,
        )
    }
}
