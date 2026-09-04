// Root build file: plugin version declarations only (apply false everywhere)
// so each module opts in to exactly what it needs. No shared config lives
// here beyond that -- see TierDependencyTest for the enforcement of "which
// module may depend on what."
plugins {
    alias(libs.plugins.android.application) apply false
    alias(libs.plugins.kotlin.jvm) apply false
    alias(libs.plugins.kotlin.compose) apply false
}
