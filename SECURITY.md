# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 1.x     | Yes                |
| < 1.0   | No                 |

## Reporting a Vulnerability

If you discover a security vulnerability within Trade Intelligence Graph, please report it responsibly.

**Do NOT open a public GitHub issue for security vulnerabilities.**

Instead, please send a detailed report to the repository maintainer via GitHub's private vulnerability reporting feature or direct communication.

### What to include:

- A description of the vulnerability
- Steps to reproduce the issue
- Potential impact assessment
- Suggested fix (if any)

### Response timeline:

- **Acknowledgment:** Within 48 hours
- **Initial assessment:** Within 5 business days
- **Fix timeline:** Depends on severity (critical: 72 hours, high: 1 week, medium: 2 weeks)

## Security Considerations

### Data Handling

This system processes trade declaration data that may contain:
- Personally identifiable information (PII) of trade entities
- Commercially sensitive transaction details
- Tax identification numbers

All data must be handled in accordance with applicable data protection regulations.

### API Security

- The GraphQL API should be deployed behind an authentication proxy in production
- CORS is restricted by configuration
- Rate limiting should be applied at the infrastructure level

### Neo4j Security

- Change the default Neo4j password immediately upon deployment
- Use encrypted Bolt connections in production
- Restrict network access to the Neo4j ports
- Enable Neo4j audit logging

### Graph Data

- Risk scores and fraud indicators are probabilistic, not deterministic
- All flagged entities require human review before enforcement action
- Graph analytics results should be treated as intelligence leads, not evidence
