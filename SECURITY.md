# Security

Please report security issues privately to the repository owner instead of
opening a public issue with exploit details.

The built-in HTTP server is intended for localhost or a trusted LAN. Binding
to a non-loopback address requires `H3_SERVE_API_KEY`, but the service does not
provide TLS, per-user authorization, billing, or tenant isolation. Internet
deployments must use a TLS reverse proxy, rate limiting, authentication, and
network-level access control.

Never commit `.env.local`, API keys, generated media, user uploads, latent
checkpoints, model weights, or runtime caches. They are excluded by `.gitignore`.
