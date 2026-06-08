# Securing your endpoint

When you `modal deploy` a serve script, Modal gives it a public `*.modal.run`
URL. Public means public: anyone who learns the URL can send requests and spend
your GPU budget. The serve scripts ship with **no authentication** because the
right control depends on how you're using the endpoint, and that's your decision
to make, not a default to inherit.

Here are the options, from least to most code. Pick one before you leave an
endpoint running.

## 1. Modal proxy auth tokens (recommended for "not wide open")

Modal can require a token on the endpoint and enforce it at the edge, before your
container ever sees the request. Add `requires_proxy_auth=True` to the web
decorator:

```python
@modal.web_server(port=8000, startup_timeout=60 * 20, requires_proxy_auth=True)
def serve(self) -> None:
    pass
```

Create a proxy auth token in your Modal workspace, then send its id and secret on
every request:

```bash
curl $URL/v1/chat/completions \
  -H 'Modal-Key: wk-...' \
  -H 'Modal-Secret: ws-...' \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemma-4-12b-it","messages":[{"role":"user","content":"hi"}]}'
```

Most OpenAI clients let you set extra headers, so a coding harness can carry
these. Docs: <https://modal.com/docs/guide/webhook-proxy-auth>.

## 2. An API key checked by the server

`build_serve_cmd()` has an optional `api_key_env` hook. Point it at an
environment variable, inject the value with a Modal Secret, and SGLang/vLLM will
reject requests without a matching `Authorization: Bearer` header. This puts the
check in the model server rather than at Modal's edge, which is handy if you want
the same key to work when you run off Modal too.

```bash
modal secret create llm-endpoint-key ENDPOINT_API_KEY=$(openssl rand -hex 32)
```

Then in the serve script, pass `api_key_env="ENDPOINT_API_KEY"` to
`build_serve_cmd(...)` and add `secrets=[modal.Secret.from_name("llm-endpoint-key")]`
to the `@app.cls`/`@app.function` decorator.

## 3. Keep it off the public internet

If you don't want a public URL at all, don't expose one. Run the container on a
private network you control — a VPN, WireGuard, or Tailscale tailnet — and reach
it by its private address. This repo doesn't ship any private-network code; it's
yours to set up, and it keeps your network topology out of a public repository.
Tailscale's own guide is a reasonable starting point:
<https://tailscale.com/kb/>.

## A few rules either way

- Never commit a token, key, or `.env`. `.gitignore` already excludes `.env*`,
  but check `git status` before your first push.
- Rotate keys you've pasted into a shell or a config file.
- Stop idle apps (`modal app stop <app-name>`). An endpoint that isn't running
  can't be abused, and it isn't billing you.

Modal's endpoint and auth docs: <https://modal.com/docs/guide/webhooks>.
