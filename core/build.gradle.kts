// :core -- the ONLY place business rules live (v3 PRD Sec10.5). Plain Kotlin/JVM,
// NOT com.android.library. That is not a placeholder to "fill in Android
// support later" -- it is the whole architecture: this module compiles
// once, and an Android app module (:android:app) and a desktop CLI (:edge)
// both depend on the SAME compiled output directly. If this module ever
// needs `com.android.library` applied, that is an architecture violation,
// not a convenience fix -- see TierDependencyTest.
plugins {
    alias(libs.plugins.kotlin.jvm)
}

kotlin {
    jvmToolchain(17)
    compilerOptions {
        // Bytecode target must work for BOTH desktop JDK 17 (:edge) and
        // Android's D8/R8 ingestion (:android:app depending on this module).
        // AGP 9.3.0 (current, TOOLCHAIN.md) accepts JVM_17 bytecode from a
        // library dependency without desugaring gymnastics.
        jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
        freeCompilerArgs.add("-Xjsr305=strict")
    }
}

dependencies {
    implementation(libs.kotlinx.coroutines.core)
    // Small, dependency-free JSON reader for ModelManifest/ReplaySchema --
    // chosen over hand-rolled regex parsing once the manifest shape gets
    // nested objects and mixed types (see ModelManifest.kt doc comment for
    // why the CSV schema sync test could stay regex-based but this could not).
    implementation(libs.org.json)

    // ONNX Runtime: compileOnly here, deliberately. :core's ModelRunner.kt
    // is written against the ai.onnxruntime.* API surface, which is
    // identical between the desktop (`onnxruntime`) and Android
    // (`onnxruntime-android`) artifacts -- but bundling either one's native
    // .so/.dll here would make :core's "same bytecode, either target" claim
    // false (native libs are target-specific, Java bytecode is not). Each
    // leaf module (:android:app, :edge) supplies the concrete runtime.
    compileOnly(libs.onnxruntime.desktop)

    testImplementation(libs.junit.jupiter.api)
    testImplementation(libs.junit.jupiter.params)
    testRuntimeOnly(libs.junit.jupiter.engine)
    testImplementation(libs.archunit.junit5)
    // Desktop ONNX Runtime for tests ONLY -- this is what lets the
    // golden-vector test (Sec2.3/Sec7.7 of PRD-ANDROID-ENGINE.md) run as a
    // plain JUnit test on a laptop, no emulator needed. Does not leak into
    // what :android:app or :edge ship, since testImplementation is test-
    // source-set-only.
    testImplementation(libs.onnxruntime.desktop)
}

tasks.test {
    useJUnitPlatform()
    // ArchUnit needs full classpath visibility to scan for forbidden
    // android.*/androidx.* references -- default test task already has this,
    // stated explicitly so it's not accidentally narrowed later.
    testLogging {
        events("passed", "skipped", "failed")
    }
}
