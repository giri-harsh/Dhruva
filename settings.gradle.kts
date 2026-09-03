// Module map, per PRD-ANDROID-ENGINE.md Sec5.4 and v3 PRD Sec10.5, adapted:
// :core has ZERO Android dependencies (plain Kotlin/JVM) so the identical
// compiled bytecode is consumed by both an Android app and a desktop CLI --
// this is the entire mechanism behind "one artefact, three consumers"
// (v3 PRD Sec10.2). :core is NOT `com.android.library` -- if it ever needs
// one, that is itself a violation of the architecture and should fail
// TierDependencyTest, not be quietly added here.
pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "dhruva"

include(":core")
include(":android:app")
include(":edge")
