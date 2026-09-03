package org.anchor.archunit

import com.tngtech.archunit.core.importer.ClassFileImporter
import com.tngtech.archunit.lang.syntax.ArchRuleDefinition.noClasses
import org.junit.jupiter.api.Test

/**
 * v3 PRD Section10.1's dependency rule, made mechanical: ":core is the ONLY
 * place business rules live" and must never depend on anything Android-
 * specific, so the exact same compiled bytecode genuinely IS what both
 * :android:app and :edge consume (v3 PRD Section10.2's whole architecture
 * claim rests on this).
 *
 * This is deliberately belt-and-suspenders on top of an already-real
 * guarantee: core/build.gradle.kts adds no android/androidx dependency at
 * all, so an accidental `import android.hardware.Sensor` inside
 * org.anchor.* would already fail to COMPILE (the symbol would not
 * resolve) before this test ever runs. What this test additionally
 * catches is the scenario the compile-time guarantee cannot: someone adds
 * an android/androidx dependency to core/build.gradle.kts later "just to
 * unblock something," making such an import compile-legal again -- this
 * test fails immediately the day that happens, rather than silently
 * letting the architecture erode.
 */
class TierDependencyTest {

    @Test
    fun `org anchor classes never depend on android or androidx packages`() {
        val classes = ClassFileImporter().importPackages("org.anchor")

        val rule = noClasses()
            .that().resideInAPackage("org.anchor..")
            .should().dependOnClassesThat().resideInAnyPackage("android..", "androidx..")

        rule.check(classes)
    }
}
