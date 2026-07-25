import { describe, expect, it } from "vitest";
import tauriConf from "../src-tauri/tauri.conf.json";
import rightRailSource from "./components/RightRail.tsx?raw";

// #99: the shell shipped with `csp: null`, so the webview ran with no Content Security Policy at
// all. Tauri only emits the header when the config sets one, and the policy is inherited by the
// artifact `srcdoc` iframe (RightRail.tsx), so a regression here silently re-opens the
// XSS -> unauthenticated local sidecar API pivot.
//
// These are config/source invariants, deliberately. The real header only exists on the `tauri://`
// origin of a packaged build: it is absent under `tauri dev` (the devUrl path returns no CSP) and
// invisible to both vitest (jsdom) and Playwright (vite dev server). This file is the only
// automated signal available; the end-to-end check is a packaged-build walkthrough.

const csp = (
  tauriConf as unknown as { app: { security: { csp: Record<string, string> | null } } }
).app.security.csp;

const tokensOf = (directive: string): string[] =>
  (csp?.[directive] ?? "").split(/\s+/).filter(Boolean);

// Each directive and the sources it must allow, traced to real call sites:
//   connect-src  ipc:/http://ipc.localhost  -> the invokes in src/tauri.ts (folder picker,
//                autostart, keep-awake, dictation); loopback wildcards -> the sidecar port, which
//                src-tauri/src/lib.rs `free_port()` re-picks every launch.
//   img-src      data: -> data_url attachment previews and Vite-inlined provider logos;
//                https: -> remote images in agent-authored markdown.
//   style-src    'unsafe-inline' -> React `style={{…}}` attributes, which cannot take a nonce.
//   worker-src   the bundled pdf.js worker.
const REQUIRED: ReadonlyArray<readonly [string, readonly string[]]> = [
  ["default-src", ["'self'"]],
  ["script-src", ["'self'"]],
  ["style-src", ["'self'", "'unsafe-inline'"]],
  ["img-src", ["'self'", "data:", "https:"]],
  ["font-src", ["'self'"]],
  ["worker-src", ["'self'"]],
  [
    "connect-src",
    ["'self'", "ipc:", "http://ipc.localhost", "http://127.0.0.1:*", "ws://127.0.0.1:*"],
  ],
  ["frame-src", ["'self'"]],
  ["object-src", ["'none'"]],
  ["base-uri", ["'self'"]],
  ["form-action", ["'none'"]],
];

describe("desktop CSP", () => {
  it("is configured at all (the #99 regression: csp was null)", () => {
    expect(csp).toBeTruthy();
    expect(typeof csp).toBe("object");
  });

  it.each(REQUIRED)("%s allows the sources it needs", (directive, sources) => {
    const actual = tokensOf(directive);
    for (const source of sources) expect(actual).toContain(source);
  });

  it("pins default-src to 'self' exactly", () => {
    expect(tokensOf("default-src")).toEqual(["'self'"]);
  });

  // Forbidden sources. `'unsafe-eval'` is why RightRail passes isEvalSupported: false to pdf.js,
  // and `'unsafe-inline'` in script-src would defeat the point of the policy — Tauri hashes the
  // inline theme script in index.html at build time, so it is never needed.
  it.each([
    ["'unsafe-eval'", "any directive"],
    ["'unsafe-inline'", "script-src"],
  ])("never allows %s in %s", (token, scope) => {
    const directives = scope === "script-src" ? ["script-src"] : Object.keys(csp ?? {});
    for (const directive of directives) expect(tokensOf(directive)).not.toContain(token);
  });

  // Edge case: a bare `*` is a wildcard host and must never appear, but the loopback entries
  // legitimately end in `:*` (a port wildcard). Token-exact matching, not substring, is what keeps
  // `http://127.0.0.1:*` passing while `*` fails.
  it("has no bare wildcard source in any directive", () => {
    for (const directive of Object.keys(csp ?? {})) {
      expect(tokensOf(directive)).not.toContain("*");
    }
  });

  it("keeps the port wildcard on the loopback sidecar origins", () => {
    expect(tokensOf("connect-src")).toEqual(
      expect.arrayContaining(["http://127.0.0.1:*", "ws://127.0.0.1:*"]),
    );
  });
});

describe("pdf.js needs no unsafe-eval", () => {
  // pdf.js defaults isEvalSupported to true and compiles PostScript functions with `new Function()`
  // in worker code, which inherits this document's policy — so the default would demand
  // 'unsafe-eval'. Asserts the call site, not runtime pdf.js behaviour: without this, doing the
  // config change and forgetting the pdf.js one leaves the suite green while packaged PDF
  // artifacts break, in exactly the blind spot no test here can reach.
  it("passes isEvalSupported: false at the getDocument call site", () => {
    expect(rightRailSource).toMatch(/getDocument\(\{[^}]*isEvalSupported:\s*false[^}]*\}\)/);
  });
});
