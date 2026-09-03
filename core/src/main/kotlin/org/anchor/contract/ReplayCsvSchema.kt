package org.anchor.contract

/**
 * The frozen contracts/replay_csv/schema.json, restated as a compile-time
 * checked Kotlin constant instead of parsed at runtime. Deliberate choice
 * over a generic JSON-schema-driven parser: this is a small, fixed,
 * rarely-changing 15-column contract, and a hardcoded list gives real type
 * safety (CsvReplaySchemaSyncTest fails loudly the moment schema.json's
 * committed column list diverges from this one -- see that test for the
 * mechanism). "Centralise contract interpretation so schema changes cannot
 * silently create inconsistencies" (PRD-ANDROID-ENGINE.md Sec7.2) is
 * satisfied by that test, not by parsing JSON at every startup.
 *
 * contracts/replay_csv/ ownership is currently unclear between the two
 * tracks (PRD-ANDROID-ENGINE.md Sec2.1/Sec2.7 say Kamal owns it; explicit
 * instruction received this session says treat all frozen contracts as
 * read-only unless absolutely necessary) -- this file does not modify
 * schema.json, only mirrors it with a synchronisation test as the tripwire.
 */
object ReplayCsvSchema {
    const val CONTRACT_VERSION = "1.0.0"

    /** Exact order. Nothing downstream infers a column from its name alone. */
    val COLUMNS: List<String> = listOf(
        "timestamp_ms",
        "accel_x", "accel_y", "accel_z",
        "gyro_x", "gyro_y", "gyro_z",
        "mag_x", "mag_y", "mag_z",
        "gnss_valid",
        "gnss_lat", "gnss_lon", "gnss_speed_mps", "gnss_course_deg",
    )

    /** Columns that may be the empty string. Everything else must never be empty. */
    val NULLABLE_COLUMNS: Set<String> = setOf(
        "gnss_lat", "gnss_lon", "gnss_speed_mps", "gnss_course_deg",
    )

    const val GNSS_INDEX_VALID = 10
    const val GNSS_INDEX_LAT = 11
    const val GNSS_INDEX_LON = 12
    const val GNSS_INDEX_SPEED = 13
    const val GNSS_INDEX_COURSE = 14
}
