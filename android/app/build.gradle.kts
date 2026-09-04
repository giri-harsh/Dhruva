// Tier 1 (Presentation) only -- v3 PRD Sec10.0. No business rule, no
// repository access lives in this module; it renders what :core hands it
// and forwards intents back. Enforced by TierDependencyTest in :core.
plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.compose)
}

android {
    namespace = "org.anchor.android"
    // See TOOLCHAIN.md's Android table for the reasoning behind every
    // number below -- this is not "pick a default," each is a decision.
    compileSdk = 37

    defaultConfig {
        applicationId = "org.anchor.android"
        minSdk = 29   // fixed by v3 PRD A4 -- not a build-file choice
        targetSdk = 36
        versionCode = 1
        versionName = "0.1.0-week1"

        // Restrict ABIs to keep APK size down; :core/:edge don't bundle
        // native libs at all, so this only affects onnxruntime-android's
        // .so payload here. x86_64 kept for the emulator only.
        ndk {
            abiFilters += listOf("arm64-v8a", "x86_64")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false // revisit once there's real UI to shrink
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    buildFeatures {
        compose = true
    }
}

dependencies {
    // The whole engine -- alignment, filter, model adapter, replay,
    // everything under Sec4/Sec5 of PRD-ANDROID-ENGINE.md -- lives in
    // :core. This module renders it.
    implementation(project(":core"))

    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.activity.compose)
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.compose.ui)
    implementation(libs.androidx.compose.ui.graphics)
    implementation(libs.androidx.compose.ui.tooling.preview)
    implementation(libs.androidx.compose.material3)

    // The mobile ONNX Runtime AAR -- native binaries for the ABIs above.
    // :core references ai.onnxruntime.* at compileOnly scope; this is what
    // actually supplies the implementation at runtime on-device.
    implementation(libs.onnxruntime.android)

    testImplementation(libs.junit.jupiter.api)
    testRuntimeOnly(libs.junit.jupiter.engine)
    // androidTestImplementation deliberately NOT added yet -- the
    // instrumented golden-vector test (android-contract-check in
    // contracts-ci.yml) needs a connected device/emulator to actually run,
    // which this dev environment does not have. Scaffolded as a Week-1
    // to-build item, not yet source here -- see the Week-1 report.
}
