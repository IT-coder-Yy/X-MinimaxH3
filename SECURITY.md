# Security policy

## Supported version

Security fixes are applied to the current `0.7.x` release line. Research
calibration folders and older extracted prototypes are not supported products.

## Reporting a vulnerability

Do not publish credentials, private prompts, reference media or an exploit in a
public issue. Use the repository's private GitHub Security Advisory channel and
include the affected version, operating system, reproduction steps and impact.

## Deployment boundary

The default listener is `127.0.0.1:8090`. When binding to a non-loopback
address, X-MinimaxH3 refuses to start unless `H3_SERVE_API_KEY` is configured.
Operators remain responsible for host firewalling, TLS termination, user access
control and the safeguards required by the MiniMax H3 Community License.

MiMo and service API keys are local secrets. They are excluded from Git and
must never be included in issue reports, workspaces or example workflows.
