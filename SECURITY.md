# Security Policy

## Supported versions

Security fixes target the latest `main` branch. This repository is a
teaching-sized demonstration, not a production ad server. Use synthetic data
and do not expose a demonstration deployment to untrusted networks.

## Reporting a vulnerability

Report suspected vulnerabilities privately through
[GitHub private vulnerability reporting](https://github.com/matrixarkai/MatrixAdsDemo/security/advisories/new).
Do not publish exploit details in issues or pull requests before coordinated
disclosure.

Include the affected commit, component, reproduction steps, and potential impact.
Do not include credentials, personal data, or private advertising data.

## Deployment boundaries

Do not load index artifacts from untrusted sources: the demonstration uses
Python pickle, which can execute code during deserialization. Build artifacts
from a trusted checkout and keep them under the operator's control.
Keep credentials and real user data out of source control and container images.
