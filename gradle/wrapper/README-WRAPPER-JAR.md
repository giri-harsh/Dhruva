# Wrapper scripts (gradlew / gradlew.bat) and gradle-wrapper.jar: not committed yet

`gradle-wrapper.properties` in this directory IS committed and correct --
it pins Gradle 9.5.0 with its official sha256 (fetched from
services.gradle.org, matching TOOLCHAIN.md's Android table).

`gradlew`, `gradlew.bat`, and `gradle-wrapper.jar` are NOT committed. This
dev environment has no `gradle` binary, so I can't generate the real ones
-- and hand-transcribing ~200 lines of vendor bootstrap script from memory
into a text file is exactly the kind of thing that introduces a silent,
hard-to-spot bug (a dropped line, a wrong escape) that then sits there
looking legitimate. The correct tool for generating these losslessly
already exists and is one command away for whoever has real tooling first.

**Whoever has real Gradle/Android Studio tooling first, run this once
from the repo root, then commit the result:**

```bash
gradle wrapper --gradle-version 9.5.0 --distribution-type bin
```

This generates all three files together, correctly, from Gradle itself --
and will not change `gradle-wrapper.properties`, since ours already
matches what that command would produce. After that, everyone else's
`./gradlew` works with zero local Gradle install, which is the entire
point of the wrapper. Delete this file once all three are committed.
