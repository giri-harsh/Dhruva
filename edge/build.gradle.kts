// Edge/CLI engine (FR-21) -- CSV or 200 Hz serial IMU in, pose stream out.
// The point of this module's existence is that it is almost nothing: it
// depends on :core exactly like :android:app does, supplies the DESKTOP
// ONNX Runtime artifact instead of the mobile one, and adds a thin
// SerialImuSource + a terminal entry point. If this module starts
// accumulating its own filter/alignment/model logic instead of using
// :core's, that's the architecture breaking, not a feature.
plugins {
    alias(libs.plugins.kotlin.jvm)
    application
}

kotlin {
    jvmToolchain(17)
}

application {
    mainClass.set("org.anchor.edge.MainKt")
}

dependencies {
    implementation(project(":core"))
    implementation(libs.onnxruntime.desktop)

    testImplementation(libs.junit.jupiter.api)
    testRuntimeOnly(libs.junit.jupiter.engine)
}

tasks.test {
    useJUnitPlatform()
}
