#!/usr/bin/env python3
"""
Production Security Audit System
================================

Comprehensive security auditing for SolTrader production deployment:
- Wallet key security validation
- API key management and rotation
- Network security assessment
- Dependency vulnerability scanning
- Configuration security review

Security audit features:
- Automated security scanning
- Vulnerability detection and reporting
- Best practices compliance checking
- Security configuration validation
- Risk assessment and recommendations
"""

import os
import re
import hashlib
import hmac
import secrets
import json
import logging
import asyncio
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import base64
import urllib.parse

logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    """Security risk levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class VulnerabilityType(Enum):
    """Types of security vulnerabilities"""
    CREDENTIAL_EXPOSURE = "credential_exposure"
    WEAK_ENCRYPTION = "weak_encryption"
    INSECURE_CONFIG = "insecure_config"
    DEPENDENCY_VULN = "dependency_vulnerability"
    NETWORK_EXPOSURE = "network_exposure"
    ACCESS_CONTROL = "access_control"
    DATA_LEAKAGE = "data_leakage"

@dataclass
class SecurityFinding:
    """Security audit finding"""
    id: str
    title: str
    description: str
    severity: SecurityLevel
    vulnerability_type: VulnerabilityType
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    evidence: Optional[str] = None
    recommendation: str = ""
    cwe_id: Optional[str] = None  # Common Weakness Enumeration
    cvss_score: Optional[float] = None

@dataclass
class SecurityAuditReport:
    """Complete security audit report"""
    audit_id: str
    timestamp: datetime
    findings: List[SecurityFinding]
    summary: Dict[str, Any]
    recommendations: List[str]
    compliance_status: Dict[str, bool]
    risk_score: float

class WalletSecurityValidator:
    """Validate wallet key security"""
    
    def __init__(self):
        self.sensitive_patterns = [
            r'[0-9a-fA-F]{64}',  # Private keys (64 hex chars)
            r'[0-9a-fA-F]{128}', # Extended private keys
            r'[A-Za-z0-9]{88}',  # Base58 encoded keys (typical Solana)
            r'[A-Za-z0-9+/=]{44}', # Base64 encoded keys
        ]
        
        logger.info("WalletSecurityValidator initialized")
    
    async def scan_for_exposed_keys(self, directory: str) -> List[SecurityFinding]:
        """Scan for exposed private keys in files"""
        
        findings = []
        exclude_patterns = ['.git', '__pycache__', 'node_modules', '.env.template']
        
        try:
            for root, dirs, files in os.walk(directory):
                # Skip excluded directories
                dirs[:] = [d for d in dirs if not any(pattern in d for pattern in exclude_patterns)]
                
                for file in files:
                    if file.endswith(('.py', '.js', '.json', '.env', '.txt', '.md', '.yml', '.yaml')):
                        file_path = os.path.join(root, file)
                        file_findings = await self._scan_file_for_keys(file_path)
                        findings.extend(file_findings)
        
        except Exception as e:
            logger.error(f"Error scanning for exposed keys: {e}")
            findings.append(SecurityFinding(
                id="wallet_scan_error",
                title="Wallet Scan Error",
                description=f"Failed to complete wallet key scan: {str(e)}",
                severity=SecurityLevel.MEDIUM,
                vulnerability_type=VulnerabilityType.CREDENTIAL_EXPOSURE,
                recommendation="Investigate scan failure and ensure proper file permissions"
            ))
        
        return findings
    
    async def _scan_file_for_keys(self, file_path: str) -> List[SecurityFinding]:
        """Scan individual file for potential private keys"""
        
        findings = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
                
                for line_num, line in enumerate(lines, 1):
                    for pattern in self.sensitive_patterns:
                        matches = re.finditer(pattern, line)
                        for match in matches:
                            # Skip if it's clearly a comment or example
                            if self._is_false_positive(line, match.group()):
                                continue
                            
                            findings.append(SecurityFinding(
                                id=f"exposed_key_{hashlib.md5((file_path + str(line_num)).encode()).hexdigest()[:8]}",
                                title="Potential Private Key Exposure",
                                description=f"Potential private key or sensitive data found in file",
                                severity=SecurityLevel.CRITICAL,
                                vulnerability_type=VulnerabilityType.CREDENTIAL_EXPOSURE,
                                file_path=file_path,
                                line_number=line_num,
                                evidence=f"Pattern match: {match.group()[:20]}...",
                                recommendation="Remove private keys from source code. Use environment variables or secure key management.",
                                cwe_id="CWE-798"
                            ))
        
        except Exception as e:
            logger.warning(f"Could not scan file {file_path}: {e}")
        
        return findings
    
    def _is_false_positive(self, line: str, match: str) -> bool:
        """Check if match is likely a false positive"""
        
        line_lower = line.lower()
        
        # Skip comments and documentation
        if any(marker in line_lower for marker in ['#', '//', '/*', 'example', 'test', 'mock', 'dummy']):
            return True
        
        # Skip if it's clearly a placeholder
        if any(placeholder in match.lower() for placeholder in ['your_', 'test_', 'example_', 'dummy_']):
            return True
        
        # Skip if it's all zeros or ones (likely placeholder)
        if match in ['0' * len(match), '1' * len(match)]:
            return True
        
        return False
    
    def validate_key_storage(self, config: Dict[str, Any]) -> List[SecurityFinding]:
        """Validate how keys are stored and accessed"""
        
        findings = []
        
        # Check if private keys are in environment variables
        sensitive_env_vars = [
            'WALLET_PRIVATE_KEY', 'PRIVATE_KEY', 'SECRET_KEY', 'API_SECRET'
        ]
        
        for var in sensitive_env_vars:
            if var in os.environ:
                value = os.environ[var]
                
                if len(value) > 32 and not value.startswith('encrypted:'):
                    findings.append(SecurityFinding(
                        id=f"unencrypted_env_{var.lower()}",
                        title="Unencrypted Sensitive Data in Environment",
                        description=f"Environment variable {var} contains unencrypted sensitive data",
                        severity=SecurityLevel.HIGH,
                        vulnerability_type=VulnerabilityType.CREDENTIAL_EXPOSURE,
                        recommendation="Encrypt sensitive environment variables or use secure key management service",
                        cwe_id="CWE-312"
                    ))
        
        return findings

class APISecurityValidator:
    """Validate API security and key management"""
    
    def __init__(self):
        self.api_key_patterns = [
            r'sk-[a-zA-Z0-9]{48}',  # OpenAI style
            r'[a-zA-Z0-9]{32,}',    # Generic API keys
            r'Bearer [a-zA-Z0-9+/=]+', # Bearer tokens
        ]
        
        logger.info("APISecurityValidator initialized")
    
    async def audit_api_security(self, config_files: List[str]) -> List[SecurityFinding]:
        """Audit API security configuration"""
        
        findings = []
        
        # Check API key rotation
        findings.extend(self._check_api_key_rotation())
        
        # Check API rate limiting
        findings.extend(self._check_rate_limiting())
        
        # Check API authentication
        findings.extend(self._check_api_authentication())
        
        # Scan for exposed API keys
        for config_file in config_files:
            if os.path.exists(config_file):
                file_findings = await self._scan_for_api_keys(config_file)
                findings.extend(file_findings)
        
        return findings
    
    def _check_api_key_rotation(self) -> List[SecurityFinding]:
        """Check API key rotation configuration"""
        
        findings = []
        
        rotation_enabled = os.getenv('ENABLE_API_KEY_ROTATION', 'false').lower() == 'true'
        rotation_hours = int(os.getenv('API_KEY_ROTATION_HOURS', '0'))
        
        if not rotation_enabled:
            findings.append(SecurityFinding(
                id="api_key_rotation_disabled",
                title="API Key Rotation Disabled",
                description="API key rotation is not enabled",
                severity=SecurityLevel.MEDIUM,
                vulnerability_type=VulnerabilityType.ACCESS_CONTROL,
                recommendation="Enable API key rotation to limit exposure from compromised keys",
                cwe_id="CWE-324"
            ))
        elif rotation_hours > 168:  # More than 7 days
            findings.append(SecurityFinding(
                id="api_key_rotation_infrequent",
                title="Infrequent API Key Rotation",
                description=f"API keys rotate every {rotation_hours} hours (recommended: ≤168 hours)",
                severity=SecurityLevel.LOW,
                vulnerability_type=VulnerabilityType.ACCESS_CONTROL,
                recommendation="Consider more frequent API key rotation for better security"
            ))
        
        return findings
    
    def _check_rate_limiting(self) -> List[SecurityFinding]:
        """Check API rate limiting configuration"""
        
        findings = []
        
        rate_limit = int(os.getenv('API_RATE_LIMIT_PER_MINUTE', '0'))
        
        if rate_limit == 0:
            findings.append(SecurityFinding(
                id="no_rate_limiting",
                title="No API Rate Limiting",
                description="API rate limiting is not configured",
                severity=SecurityLevel.MEDIUM,
                vulnerability_type=VulnerabilityType.NETWORK_EXPOSURE,
                recommendation="Configure API rate limiting to prevent abuse",
                cwe_id="CWE-770"
            ))
        elif rate_limit > 10000:
            findings.append(SecurityFinding(
                id="high_rate_limit",
                title="High API Rate Limit",
                description=f"API rate limit is very high ({rate_limit}/minute)",
                severity=SecurityLevel.LOW,
                vulnerability_type=VulnerabilityType.NETWORK_EXPOSURE,
                recommendation="Consider lower rate limits to prevent abuse while maintaining functionality"
            ))
        
        return findings
    
    def _check_api_authentication(self) -> List[SecurityFinding]:
        """Check API authentication configuration"""
        
        findings = []
        
        # Check if CORS is properly configured
        cors_enabled = os.getenv('ENABLE_CORS', 'true').lower() == 'true'
        allowed_origins = os.getenv('CORS_ALLOWED_ORIGINS', '')
        
        if cors_enabled and not allowed_origins:
            findings.append(SecurityFinding(
                id="cors_misconfiguration",
                title="CORS Misconfiguration",
                description="CORS is enabled but no allowed origins specified",
                severity=SecurityLevel.MEDIUM,
                vulnerability_type=VulnerabilityType.NETWORK_EXPOSURE,
                recommendation="Configure specific allowed origins for CORS or disable if not needed",
                cwe_id="CWE-346"
            ))
        
        return findings
    
    async def _scan_for_api_keys(self, file_path: str) -> List[SecurityFinding]:
        """Scan file for exposed API keys"""
        
        findings = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
                
                for line_num, line in enumerate(lines, 1):
                    for pattern in self.api_key_patterns:
                        matches = re.finditer(pattern, line)
                        for match in matches:
                            if not self._is_api_key_false_positive(line, match.group()):
                                findings.append(SecurityFinding(
                                    id=f"exposed_api_key_{hashlib.md5((file_path + str(line_num)).encode()).hexdigest()[:8]}",
                                    title="Potential API Key Exposure",
                                    description="Potential API key found in configuration file",
                                    severity=SecurityLevel.HIGH,
                                    vulnerability_type=VulnerabilityType.CREDENTIAL_EXPOSURE,
                                    file_path=file_path,
                                    line_number=line_num,
                                    evidence=f"Pattern match: {match.group()[:20]}...",
                                    recommendation="Remove API keys from configuration files. Use environment variables.",
                                    cwe_id="CWE-798"
                                ))
        
        except Exception as e:
            logger.warning(f"Could not scan file {file_path} for API keys: {e}")
        
        return findings
    
    def _is_api_key_false_positive(self, line: str, match: str) -> bool:
        """Check if API key match is likely a false positive"""
        
        line_lower = line.lower()
        
        # Skip comments and examples
        if any(marker in line_lower for marker in ['#', '//', 'example', 'your_', 'test_']):
            return True
        
        # Skip if it's clearly a placeholder
        if any(placeholder in match.lower() for placeholder in ['your_', 'test_', 'example_']):
            return True
        
        return False

class NetworkSecurityValidator:
    """Validate network security configuration"""
    
    def __init__(self):
        logger.info("NetworkSecurityValidator initialized")
    
    async def audit_network_security(self) -> List[SecurityFinding]:
        """Audit network security configuration"""
        
        findings = []
        
        # Check allowed hosts configuration
        findings.extend(self._check_allowed_hosts())
        
        # Check SSL/TLS configuration
        findings.extend(self._check_ssl_configuration())
        
        # Check exposed ports
        findings.extend(await self._check_exposed_ports())
        
        # Check firewall configuration
        findings.extend(self._check_firewall_config())
        
        return findings
    
    def _check_allowed_hosts(self) -> List[SecurityFinding]:
        """Check allowed hosts configuration"""
        
        findings = []
        
        allowed_hosts = os.getenv('ALLOWED_HOSTS', '')
        
        if not allowed_hosts:
            findings.append(SecurityFinding(
                id="no_allowed_hosts",
                title="No Allowed Hosts Configured",
                description="No allowed hosts restriction configured",
                severity=SecurityLevel.MEDIUM,
                vulnerability_type=VulnerabilityType.NETWORK_EXPOSURE,
                recommendation="Configure allowed hosts to restrict access to known domains/IPs",
                cwe_id="CWE-346"
            ))
        elif '*' in allowed_hosts:
            findings.append(SecurityFinding(
                id="wildcard_allowed_hosts",
                title="Wildcard in Allowed Hosts",
                description="Wildcard (*) found in allowed hosts configuration",
                severity=SecurityLevel.HIGH,
                vulnerability_type=VulnerabilityType.NETWORK_EXPOSURE,
                recommendation="Replace wildcard with specific allowed hosts",
                cwe_id="CWE-346"
            ))
        
        return findings
    
    def _check_ssl_configuration(self) -> List[SecurityFinding]:
        """Check SSL/TLS configuration"""
        
        findings = []
        
        # Check if HTTPS is enforced
        force_https = os.getenv('FORCE_HTTPS', 'false').lower() == 'true'
        
        if not force_https:
            findings.append(SecurityFinding(
                id="no_https_enforcement",
                title="HTTPS Not Enforced",
                description="HTTPS enforcement is not enabled",
                severity=SecurityLevel.MEDIUM,
                vulnerability_type=VulnerabilityType.NETWORK_EXPOSURE,
                recommendation="Enable HTTPS enforcement for all communications",
                cwe_id="CWE-319"
            ))
        
        return findings
    
    async def _check_exposed_ports(self) -> List[SecurityFinding]:
        """Check for exposed ports"""
        
        findings = []
        
        try:
            # Check common ports that shouldn't be exposed
            dangerous_ports = [22, 3389, 5432, 3306, 6379, 27017]  # SSH, RDP, PostgreSQL, MySQL, Redis, MongoDB
            
            # This would normally check actual network connections
            # For now, just check configured ports
            service_port = int(os.getenv('SERVICE_PORT', '8080'))
            prometheus_port = int(os.getenv('PROMETHEUS_PORT', '8000'))
            grafana_port = int(os.getenv('GRAFANA_PORT', '3000'))
            
            configured_ports = [service_port, prometheus_port, grafana_port]
            
            for port in configured_ports:
                if port in dangerous_ports:
                    findings.append(SecurityFinding(
                        id=f"dangerous_port_{port}",
                        title=f"Dangerous Port Exposed: {port}",
                        description=f"Service configured on potentially dangerous port {port}",
                        severity=SecurityLevel.HIGH,
                        vulnerability_type=VulnerabilityType.NETWORK_EXPOSURE,
                        recommendation=f"Consider using a different port than {port} or ensure proper access controls",
                        cwe_id="CWE-200"
                    ))
        
        except Exception as e:
            logger.error(f"Error checking exposed ports: {e}")
        
        return findings
    
    def _check_firewall_config(self) -> List[SecurityFinding]:
        """Check firewall configuration"""
        
        findings = []
        
        # This is a simplified check - in production, would integrate with actual firewall APIs
        firewall_enabled = os.getenv('FIREWALL_ENABLED', 'false').lower() == 'true'
        
        if not firewall_enabled:
            findings.append(SecurityFinding(
                id="no_firewall",
                title="No Firewall Configuration",
                description="No firewall configuration detected",
                severity=SecurityLevel.HIGH,
                vulnerability_type=VulnerabilityType.NETWORK_EXPOSURE,
                recommendation="Configure firewall to restrict access to necessary ports only",
                cwe_id="CWE-200"
            ))
        
        return findings

class DependencySecurityValidator:
    """Validate dependency security"""
    
    def __init__(self):
        logger.info("DependencySecurityValidator initialized")
    
    async def audit_dependencies(self) -> List[SecurityFinding]:
        """Audit dependencies for vulnerabilities"""
        
        findings = []
        
        # Check Python dependencies
        python_findings = await self._check_python_dependencies()
        findings.extend(python_findings)
        
        # Check for outdated packages
        outdated_findings = await self._check_outdated_packages()
        findings.extend(outdated_findings)
        
        return findings
    
    async def _check_python_dependencies(self) -> List[SecurityFinding]:
        """Check Python dependencies for known vulnerabilities"""
        
        findings = []
        
        try:
            # Check if requirements file exists
            requirements_files = ['requirements.txt', 'requirements_updated.txt', 'pyproject.toml']
            
            for req_file in requirements_files:
                if os.path.exists(req_file):
                    findings.extend(await self._scan_requirements_file(req_file))
        
        except Exception as e:
            logger.error(f"Error checking Python dependencies: {e}")
            findings.append(SecurityFinding(
                id="dependency_scan_error",
                title="Dependency Scan Error",
                description=f"Failed to scan dependencies: {str(e)}",
                severity=SecurityLevel.MEDIUM,
                vulnerability_type=VulnerabilityType.DEPENDENCY_VULN,
                recommendation="Investigate dependency scanning failure"
            ))
        
        return findings
    
    async def _scan_requirements_file(self, file_path: str) -> List[SecurityFinding]:
        """Scan requirements file for vulnerable packages"""
        
        findings = []
        
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if line and not line.startswith('#'):
                    # Parse package name and version
                    if '==' in line:
                        package_name = line.split('==')[0].strip()
                        version = line.split('==')[1].strip()
                        
                        # Check for known vulnerable packages
                        vuln_finding = self._check_package_vulnerability(package_name, version, file_path, line_num)
                        if vuln_finding:
                            findings.append(vuln_finding)
        
        except Exception as e:
            logger.warning(f"Could not scan requirements file {file_path}: {e}")
        
        return findings
    
    def _check_package_vulnerability(self, package: str, version: str, file_path: str, line_num: int) -> Optional[SecurityFinding]:
        """Check if package version has known vulnerabilities"""
        
        # Known vulnerable packages (simplified example)
        # In production, this would integrate with CVE databases
        vulnerable_packages = {
            'requests': {'<2.20.0': 'CVE-2018-18074'},
            'urllib3': {'<1.24.2': 'CVE-2019-11324'},
            'pyyaml': {'<5.1': 'CVE-2017-18342'},
        }
        
        if package.lower() in vulnerable_packages:
            vulns = vulnerable_packages[package.lower()]
            for version_constraint, cve in vulns.items():
                # Simplified version checking
                if version_constraint.startswith('<'):
                    constraint_version = version_constraint[1:]
                    if version < constraint_version:
                        return SecurityFinding(
                            id=f"vulnerable_package_{package}_{version}",
                            title=f"Vulnerable Package: {package}",
                            description=f"Package {package} version {version} has known vulnerability {cve}",
                            severity=SecurityLevel.HIGH,
                            vulnerability_type=VulnerabilityType.DEPENDENCY_VULN,
                            file_path=file_path,
                            line_number=line_num,
                            recommendation=f"Update {package} to version {constraint_version} or later",
                            cwe_id="CWE-937"
                        )
        
        return None
    
    async def _check_outdated_packages(self) -> List[SecurityFinding]:
        """Check for outdated packages"""
        
        findings = []
        
        try:
            # This would normally run pip list --outdated or similar
            # For now, just check common packages
            common_packages = ['requests', 'cryptography', 'urllib3']
            
            for package in common_packages:
                try:
                    # Simulate checking if package is outdated
                    # In reality, would check against PyPI or security databases
                    findings.append(SecurityFinding(
                        id=f"outdated_check_{package}",
                        title="Outdated Package Check",
                        description=f"Consider checking if {package} is up to date",
                        severity=SecurityLevel.LOW,
                        vulnerability_type=VulnerabilityType.DEPENDENCY_VULN,
                        recommendation=f"Regularly update {package} to latest secure version"
                    ))
                except:
                    pass  # Package not found
        
        except Exception as e:
            logger.error(f"Error checking outdated packages: {e}")
        
        return findings

class SecurityAuditor:
    """Main security auditor coordinating all security checks"""
    
    def __init__(self):
        self.wallet_validator = WalletSecurityValidator()
        self.api_validator = APISecurityValidator()
        self.network_validator = NetworkSecurityValidator()
        self.dependency_validator = DependencySecurityValidator()
        
        logger.info("SecurityAuditor initialized")
    
    async def run_comprehensive_audit(self, target_directory: str = ".") -> SecurityAuditReport:
        """Run comprehensive security audit"""
        
        audit_id = f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"Starting comprehensive security audit: {audit_id}")
        
        all_findings = []
        
        try:
            # Wallet security audit
            logger.info("Running wallet security audit...")
            wallet_findings = await self.wallet_validator.scan_for_exposed_keys(target_directory)
            wallet_findings.extend(self.wallet_validator.validate_key_storage({}))
            all_findings.extend(wallet_findings)
            
            # API security audit
            logger.info("Running API security audit...")
            config_files = ['production.env', '.env', 'config.json']
            api_findings = await self.api_validator.audit_api_security(config_files)
            all_findings.extend(api_findings)
            
            # Network security audit
            logger.info("Running network security audit...")
            network_findings = await self.network_validator.audit_network_security()
            all_findings.extend(network_findings)
            
            # Dependency security audit
            logger.info("Running dependency security audit...")
            dependency_findings = await self.dependency_validator.audit_dependencies()
            all_findings.extend(dependency_findings)
            
            # Generate summary and recommendations
            summary = self._generate_summary(all_findings)
            recommendations = self._generate_recommendations(all_findings)
            compliance_status = self._check_compliance(all_findings)
            risk_score = self._calculate_risk_score(all_findings)
            
            report = SecurityAuditReport(
                audit_id=audit_id,
                timestamp=datetime.now(),
                findings=all_findings,
                summary=summary,
                recommendations=recommendations,
                compliance_status=compliance_status,
                risk_score=risk_score
            )
            
            logger.info(f"Security audit completed: {len(all_findings)} findings, risk score: {risk_score:.1f}")
            return report
            
        except Exception as e:
            logger.error(f"Security audit failed: {e}")
            raise
    
    def _generate_summary(self, findings: List[SecurityFinding]) -> Dict[str, Any]:
        """Generate audit summary"""
        
        severity_counts = {level.value: 0 for level in SecurityLevel}
        vulnerability_counts = {vuln.value: 0 for vuln in VulnerabilityType}
        
        for finding in findings:
            severity_counts[finding.severity.value] += 1
            vulnerability_counts[finding.vulnerability_type.value] += 1
        
        return {
            "total_findings": len(findings),
            "severity_breakdown": severity_counts,
            "vulnerability_type_breakdown": vulnerability_counts,
            "critical_findings": severity_counts["critical"],
            "high_findings": severity_counts["high"],
            "files_with_issues": len(set(f.file_path for f in findings if f.file_path))
        }
    
    def _generate_recommendations(self, findings: List[SecurityFinding]) -> List[str]:
        """Generate security recommendations"""
        
        recommendations = []
        
        # Critical findings recommendations
        critical_findings = [f for f in findings if f.severity == SecurityLevel.CRITICAL]
        if critical_findings:
            recommendations.append("URGENT: Address all critical security findings immediately")
            
        # High priority recommendations
        high_findings = [f for f in findings if f.severity == SecurityLevel.HIGH]
        if high_findings:
            recommendations.append("Address high-severity security findings within 24 hours")
        
        # Specific recommendations based on finding types
        vuln_types = set(f.vulnerability_type for f in findings)
        
        if VulnerabilityType.CREDENTIAL_EXPOSURE in vuln_types:
            recommendations.append("Implement secure credential management with encryption")
        
        if VulnerabilityType.NETWORK_EXPOSURE in vuln_types:
            recommendations.append("Review and harden network security configuration")
        
        if VulnerabilityType.DEPENDENCY_VULN in vuln_types:
            recommendations.append("Update vulnerable dependencies and implement automated scanning")
        
        # General recommendations
        recommendations.extend([
            "Implement regular security audits (weekly in production)",
            "Set up automated security monitoring and alerting",
            "Create incident response procedures for security events",
            "Consider penetration testing before production deployment"
        ])
        
        return recommendations[:10]  # Limit to top 10 recommendations
    
    def _check_compliance(self, findings: List[SecurityFinding]) -> Dict[str, bool]:
        """Check compliance with security standards"""
        
        critical_count = len([f for f in findings if f.severity == SecurityLevel.CRITICAL])
        high_count = len([f for f in findings if f.severity == SecurityLevel.HIGH])
        
        return {
            "no_critical_vulnerabilities": critical_count == 0,
            "max_5_high_vulnerabilities": high_count <= 5,
            "credential_exposure_check": len([f for f in findings if f.vulnerability_type == VulnerabilityType.CREDENTIAL_EXPOSURE]) == 0,
            "network_security_baseline": len([f for f in findings if f.vulnerability_type == VulnerabilityType.NETWORK_EXPOSURE]) <= 2,
            "dependency_security_baseline": len([f for f in findings if f.vulnerability_type == VulnerabilityType.DEPENDENCY_VULN]) <= 3
        }
    
    def _calculate_risk_score(self, findings: List[SecurityFinding]) -> float:
        """Calculate overall security risk score (0-10, lower is better)"""
        
        if not findings:
            return 0.0
        
        # Weight by severity
        weights = {
            SecurityLevel.CRITICAL: 10.0,
            SecurityLevel.HIGH: 7.0,
            SecurityLevel.MEDIUM: 3.0,
            SecurityLevel.LOW: 1.0
        }
        
        total_weight = sum(weights[finding.severity] for finding in findings)
        max_possible = len(findings) * weights[SecurityLevel.CRITICAL]
        
        return min(10.0, (total_weight / max_possible) * 10.0) if max_possible > 0 else 0.0
    
    def export_report(self, report: SecurityAuditReport, format: str = "json") -> str:
        """Export security audit report"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"security_audit_report_{timestamp}.{format}"
        
        try:
            if format == "json":
                # Convert to JSON-serializable format
                report_dict = {
                    "audit_id": report.audit_id,
                    "timestamp": report.timestamp.isoformat(),
                    "summary": report.summary,
                    "risk_score": report.risk_score,
                    "compliance_status": report.compliance_status,
                    "recommendations": report.recommendations,
                    "findings": [
                        {
                            "id": f.id,
                            "title": f.title,
                            "description": f.description,
                            "severity": f.severity.value,
                            "vulnerability_type": f.vulnerability_type.value,
                            "file_path": f.file_path,
                            "line_number": f.line_number,
                            "evidence": f.evidence,
                            "recommendation": f.recommendation,
                            "cwe_id": f.cwe_id,
                            "cvss_score": f.cvss_score
                        }
                        for f in report.findings
                    ]
                }
                
                with open(filename, 'w') as f:
                    json.dump(report_dict, f, indent=2)
            
            logger.info(f"Security audit report exported to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Failed to export security report: {e}")
            raise

# Example usage and testing
if __name__ == "__main__":
    async def test_security_audit():
        """Test the security audit system"""
        
        print("Security Audit System Test")
        print("=" * 50)
        
        auditor = SecurityAuditor()
        
        try:
            # Run comprehensive audit
            print("Running comprehensive security audit...")
            report = await auditor.run_comprehensive_audit(".")
            
            print(f"\nSecurity Audit Results:")
            print(f"  Audit ID: {report.audit_id}")
            print(f"  Risk Score: {report.risk_score:.1f}/10.0")
            print(f"  Total Findings: {report.summary['total_findings']}")
            
            print(f"\nSeverity Breakdown:")
            for severity, count in report.summary['severity_breakdown'].items():
                if count > 0:
                    print(f"  {severity.upper()}: {count}")
            
            print(f"\nCompliance Status:")
            for check, passed in report.compliance_status.items():
                status = "PASS" if passed else "FAIL"
                print(f"  {check}: {status}")
            
            print(f"\nTop Recommendations:")
            for i, rec in enumerate(report.recommendations[:5], 1):
                print(f"  {i}. {rec}")
            
            if report.findings:
                print(f"\nSample Findings:")
                for finding in report.findings[:3]:
                    print(f"  - {finding.title} ({finding.severity.value})")
                    print(f"    {finding.description}")
            
            # Export report
            report_file = auditor.export_report(report)
            print(f"\nSecurity report exported to: {report_file}")
            
            print(f"\nSecurity audit test completed successfully!")
            
        except Exception as e:
            print(f"Security audit test failed: {e}")
            raise
    
    asyncio.run(test_security_audit())