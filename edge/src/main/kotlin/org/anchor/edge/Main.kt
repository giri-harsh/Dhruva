package org.anchor.edge

import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import org.anchor.sensors.CsvReplaySource
import org.anchor.sensors.SensorEvent
import java.nio.file.Path

/**
 * FR-21's edge/CLI engine, in its Week-1 shape: replays a contracts/
 * replay_csv-schema file through the exact same CsvReplaySource
 * :android:app will eventually use, via the exact same :core module --
 * this small amount of working code IS the "same compiled core, three
 * consumers" claim (v3 PRD Section10.2) made real, not yet the full FR-21
 * (a 200 Hz serial IMU source and continuous propagation are later work;
 * SerialImuSource does not exist yet).
 *
 * NOT compiled or run in this dev environment -- see the Week-1 report.
 * Usage once real tooling exists:
 *   ./gradlew :edge:run --args="path/to/replay.csv"
 */
fun main(args: Array<String>) {
    if (args.isEmpty()) {
        System.err.println("usage: edge <path-to-replay-csv>")
        return
    }

    val path = Path.of(args[0])
    val source = CsvReplaySource(path)

    val events = runBlocking { source.events().toList() }

    val accelCount = events.count { it is SensorEvent.Accelerometer }
    val gyroCount = events.count { it is SensorEvent.Gyroscope }
    val magCount = events.count { it is SensorEvent.Magnetometer }
    val gnssCount = events.count { it is SensorEvent.GnssFix }

    val anchor = source.clockAnchor()
    val firstMs = events.firstOrNull()?.let { anchor.toUnixEpochMillis(it.timestampNanos) }
    val lastMs = events.lastOrNull()?.let { anchor.toUnixEpochMillis(it.timestampNanos) }

    println("Dhruva edge replay: $path")
    println("  accel events:      $accelCount")
    println("  gyro events:       $gyroCount")
    println("  mag events:        $magCount")
    println("  gnss fix events:   $gnssCount")
    println("  first timestamp:   ${firstMs}ms (unix epoch)")
    println("  last timestamp:    ${lastMs}ms (unix epoch)")
    println()
    println("  Alignment, filtering, and model inference are not wired into")
    println("  this CLI path yet -- this proves the replay pipeline end to")
    println("  end through :core, which is Week 1 scope. See the Week-1")
    println("  report for what is built vs deferred.")
}
