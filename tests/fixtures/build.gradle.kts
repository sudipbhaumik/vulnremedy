// Sample build.gradle.kts (Kotlin DSL) for parser testing.
//
// Deliberately includes:
//   - Standard Kotlin DSL notation:        implementation("group:artifact:version")
//   - Platform/BOM import:                 implementation(platform("group:artifact:version"))
//   - String interpolation:                implementation("group:artifact:$varName")
//   - val property declarations for interpolation
//   - Multiple configurations: testImplementation, runtimeOnly, compileOnly, api, kapt
//   - Log4Shell:   log4j-core 2.14.1   → CVE-2021-44228
//   - Spring4Shell: spring-webmvc 5.3.17 → CVE-2022-22965

plugins {
    kotlin("jvm") version "1.7.10"
    id("org.springframework.boot") version "2.7.0"
}

val springBootVersion = "2.7.0"
val log4jVersion = "2.14.1"

dependencies {

    // CVE-2021-44228: Log4Shell — version from val interpolation
    implementation("org.apache.logging.log4j:log4j-core:$log4jVersion")

    // CVE-2022-22965: Spring4Shell
    implementation("org.springframework:spring-webmvc:5.3.17")

    // CVE-2022-42889: Text4Shell
    implementation("org.apache.commons:commons-text:1.9")

    // Spring Boot BOM — platform import
    implementation(platform("org.springframework.boot:spring-boot-dependencies:$springBootVersion"))

    // BOM-managed — no explicit version
    implementation("org.springframework.boot:spring-boot-starter-web")

    // api configuration
    api("org.slf4j:slf4j-api:1.7.36")

    // compileOnly
    compileOnly("org.projectlombok:lombok:1.18.24")

    // runtimeOnly
    runtimeOnly("com.h2database:h2:2.1.210")

    // testImplementation
    testImplementation("org.junit.jupiter:junit-jupiter:5.8.2")

    // kapt — Kotlin annotation processing
    kapt("org.projectlombok:lombok:1.18.24")
}
