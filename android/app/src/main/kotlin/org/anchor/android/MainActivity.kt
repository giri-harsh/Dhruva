package org.anchor.android

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp

/**
 * Tier 1 (Presentation) only -- v3 PRD Section10.0. Deliberately minimal:
 * Milestones 1-5 for this week stop at the ONNX adapter (see the reordering
 * rationale in this session own report), so there is no real map/marker/
 * confidence-ring UI (FR-17) here yet, and this Activity renders nothing
 * :core computed -- it exists to prove the :android:app module actually
 * links against :core (the dependency wiring in
 * android/app/build.gradle.kts) with a real Composable, not to demonstrate
 * a feature.
 *
 * NOT compiled or run in this dev environment -- see the Week-1 report for
 * why (no JDK 17+/Android SDK here). Correct by inspection against a
 * standard Jetpack Compose ComponentActivity shape.
 */
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            DhruvaWeek1Placeholder()
        }
    }
}

@Composable
private fun DhruvaWeek1Placeholder() {
    MaterialTheme {
        Surface(modifier = Modifier.fillMaxSize()) {
            Column(
                modifier = Modifier.fillMaxSize().padding(24.dp),
                verticalArrangement = Arrangement.Center,
                horizontalAlignment = Alignment.CenterHorizontally,
            ) {
                Text(text = stringResource(R.string.week1_placeholder))
            }
        }
    }
}
