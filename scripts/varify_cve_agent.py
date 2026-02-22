from vulnremedy.agents.scanner.agent import ScannerAgent
from vulnremedy.agents.cve_analyst.agent import CVEAnalystAgent

# Test with Spring PetClinic (known to have pom.xml)
print("Testing with Spring PetClinic...")
scanner = ScannerAgent()
scan_result = scanner.scan("spring-projects/spring-petclinic", "main")
print(f"✅ Scanner: {len(scan_result['dependencies'])} dependencies found")

if scan_result["dependencies"]:
    # Show some dependencies
    print("\nSample dependencies:")
    for dep in scan_result["dependencies"][:5]:
        print(f"  • {dep.fully_qualified_name}@{dep.version}")
    
    # Analyze for CVEs
    print("\nAnalyzing for CVEs...")
    analyst = CVEAnalystAgent()
    analysis_result = analyst.analyze(scan_result["dependencies"])
    print(f"✅ CVE Analyst: {len(analysis_result['findings'])} vulnerabilities found")
    
    # Show findings
    if analysis_result['findings']:
        print("\nFindings:")
        for finding in analysis_result['findings'][:5]:
            print(f"  • {finding.cve_id} ({finding.severity.value}) in {finding.affected_dependency.fully_qualified_name}")
    else:
        print("\n✓ No known vulnerabilities found (dependencies are up-to-date)")
else:
    print("❌ No dependencies found. Checking errors...")
    if scan_result.get("errors"):
        for error in scan_result["errors"]:
            print(f"  Error: {error}")