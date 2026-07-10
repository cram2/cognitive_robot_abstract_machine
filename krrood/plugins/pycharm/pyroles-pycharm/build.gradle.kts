import org.jetbrains.intellij.platform.gradle.TestFrameworkType

plugins {
    id("org.jetbrains.kotlin.jvm")
    id("org.jetbrains.intellij.platform")
}

group = providers.gradleProperty("pluginGroup").get()
version = providers.gradleProperty("pluginVersion").get()

// NOTE: Repositories are configured centrally in settings.gradle.kts
// (dependencyResolutionManagement), as required by the IntelliJ Platform Gradle Plugin 2.x.

dependencies {
    intellijPlatform {
        // Build against PyCharm. Since 2025.3 (build 253) the Community/Professional split was
        // retired and a single `pycharm(...)` distribution is published, so the older
        // `pycharmCommunity(...)`/`pycharmProfessional(...)` helpers no longer resolve for
        // 2025.3+. The Python support plugin "PythonCore" — which provides PyClass,
        // PyClassType, PyCustomMember, TypeEvalContext, etc. — is bundled with it.
        pycharm(providers.gradleProperty("platformVersion").get())
        bundledPlugin("PythonCore")

        // In-IDE test fixtures (BasePlatformTestCase, CodeInsightTestFixture, ...).
        testFramework(TestFrameworkType.Platform)
    }

    // BasePlatformTestCase is a JUnit 4 test case.
    testImplementation("junit:junit:4.13.2")
}

// The platform test fixtures run on the IDE's own JUnit 4 runner.
tasks.test {
    useJUnit()
}

intellijPlatform {
    pluginConfiguration {
        ideaVersion {
            sinceBuild = providers.gradleProperty("pluginSinceBuild")
            // No explicit upper bound: the Gradle plugin defaults it to the build platform's
            // branch (e.g. `261.*`). A hard far-future cap such as `999.*` is rejected/flagged
            // by the Marketplace, so widen the range by rebuilding against a newer platform.
        }
    }

    // `./gradlew publishPlugin` uploads to the JetBrains Marketplace using a permanent token
    // generated at https://plugins.jetbrains.com/author/me/tokens.
    publishing {
        token = providers.environmentVariable("PYCHARM_PUBLISH_TOKEN")
    }

    // The Marketplace recommends signed plugins. Supply the certificate chain + private key
    // (see README) through these env vars; remove this block to upload unsigned.
    signing {
        certificateChain = providers.environmentVariable("CERTIFICATE_CHAIN")
        privateKey = providers.environmentVariable("PRIVATE_KEY")
        password = providers.environmentVariable("PRIVATE_KEY_PASSWORD")
    }

    // `./gradlew verifyPlugin` runs JetBrains' Plugin Verifier against the recommended IDEs.
    pluginVerification {
        ides {
            recommended()
        }
    }
}

kotlin {
    // PyCharm 2024.3+ runs on JBR 21; build for the same target.
    jvmToolchain(21)
}

// One-shot release command: build & sign the distributable, then upload the current
// `pluginVersion` to the JetBrains Marketplace. Bump `pluginVersion` in gradle.properties first,
// export `PYCHARM_PUBLISH_TOKEN` (and the signing env vars), then run `./gradlew publishRelease`.
// A plain aggregator (no execution-time logic) keeps it compatible with the configuration cache;
// `publishPlugin` already logs the upload result.
tasks.register("publishRelease") {
    group = "publishing"
    description = "Builds, signs, and publishes the current plugin version to the JetBrains Marketplace."
    dependsOn(tasks.named("publishPlugin"))
}
