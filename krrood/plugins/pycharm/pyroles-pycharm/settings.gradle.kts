import org.jetbrains.intellij.platform.gradle.extensions.intellijPlatform

rootProject.name = "pyroles-pycharm"

pluginManagement {
    plugins {
        // Must be >= the Kotlin version PyCharm itself is built with (2026.1 ships Kotlin
        // 2.3 metadata); an older compiler cannot read the platform's class metadata.
        id("org.jetbrains.kotlin.jvm") version "2.3.20"
    }
}

plugins {
    // Auto-provisions the JDK 21 toolchain if it is not already installed.
    id("org.gradle.toolchains.foojay-resolver-convention") version "1.0.0"
    // Supplies the version for the build-script `org.jetbrains.intellij.platform` plugin.
    id("org.jetbrains.intellij.platform.settings") version "2.16.0"
}

@Suppress("UnstableApiUsage")
dependencyResolutionManagement {
    repositories {
        mavenCentral()

        // IntelliJ Platform artifacts (PyCharm, bundled plugins, test framework).
        intellijPlatform {
            defaultRepositories()
        }
    }
}
