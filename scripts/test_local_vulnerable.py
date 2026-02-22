"""Test CVE detection with known vulnerable dependencies."""

from vulnremedy.agents.scanner.parsers.maven_parser import MavenParser
from vulnremedy.agents.cve_analyst.agent import CVEAnalystAgent
from vulnremedy.models.cve import Ecosystem

# Parse the vulnerable pom.xml
with open("tests/fixtures/vulnerable_pom.xml") as f:
    content = f.read()

parser = MavenParser()
dependencies = parser.parse(content)

print(f"Parsed {len(dependencies)} dependencies")
for dep in dependencies:
    print(f"  • {dep.fully_qualified_name}@{dep.version}")

# Analyze for CVEs
print("\nAnalyzing for CVEs...")
analyst = CVEAnalystAgent()
analysis_result = analyst.analyze(dependencies)

print(f"CVE Analyst: {len(analysis_result['findings'])} vulnerabilities found")

if analysis_result['findings']:
    print("\nVulnerabilities detected:")
    for finding in analysis_result['findings']:
        print(f"  • {finding.cve_id} ({finding.severity.value})")
        print(f"    Package: {finding.affected_dependency.fully_qualified_name}")
        print(f"    Confidence: {finding.confidence.value}")
else:
    print("  Expected to find CVE-2021-44228 (Log4Shell)")
    print("This might mean:")
    print("  - ChromaDB doesn't have Log4Shell data")
    print("  - Version matching failed")
    print("  - Ecosystem filtering excluded results")