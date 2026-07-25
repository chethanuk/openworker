// Settings ▸ Models key flows on the shared provider gallery (§39 components, UX-021 page):
// bad key fails in place, a passing Test auto-saves and slides home to the gallery where the
// card wears its ✓. Providers are seeded in three states (OpenAI configured+used, Anthropic
// configured-unused, Z AI unconfigured w/ a prefilled endpoint behind the disclosure). The
// mock's /verify fails on a key containing "bad"; POST /v1/providers flips `configured`.
import { expect } from "@playwright/test";
import { test } from "./fixtures";

async function openModels(page) {
  await page.goto("/");
  await page.getByTestId("account-row").click();
  await page.getByRole("button", { name: "Settings", exact: true }).click();
  await page.getByRole("button", { name: "Models", exact: true }).click();
  await expect(page.getByTestId("set-provider-openai")).toBeVisible();
}

test("Test with a bad key fails in place; a good key saves and returns to the gallery", async ({
  page,
}) => {
  await openModels(page);
  await page.getByTestId("set-provider-zai").click();

  await page.getByTestId("set-field-api_key").fill("sk-bad-key");
  await page.getByTestId("set-test").click();
  await expect(page.getByText("Invalid API key.")).toBeVisible();

  // A good key: Test verifies AND saves (§39) — the in-field pill confirms, then the form
  // slides home and the card wears its ✓.
  await page.getByTestId("set-field-api_key").fill("sk-glm-realkey");
  await page.getByTestId("set-test").click();
  await expect(page.getByTestId("set-saved-pill")).toContainText("Tested & saved");
  await expect(page.getByTestId("set-provider-zai")).toContainText("✓ Connected", {
    timeout: 5_000,
  });

  // State-restore regression (owner catch 2026-07-19): revisiting the just-saved provider
  // must show the masked placeholder + saved pill — never the typed key restored as a draft
  // (the auto-return used to stash the saved key and replay it on the next open).
  await page.getByTestId("set-provider-zai").click();
  await expect(page.getByTestId("set-field-api_key")).toHaveValue("");
  await expect(page.getByTestId("set-field-api_key")).toHaveAttribute("placeholder", "••••••••");
  await expect(page.getByTestId("set-saved-pill")).toContainText("Tested & saved");
});

test("a configured provider's form opens with the saved state, no plaintext key", async ({
  page,
}) => {
  await openModels(page);
  await page.getByTestId("set-provider-openai").click();
  // Stored credentials show as the in-field saved pill + masked placeholder — never the key.
  await expect(page.getByTestId("set-saved-pill")).toContainText("Tested & saved");
  await expect(page.getByTestId("set-field-api_key")).toHaveValue("");
  await expect(page.getByTestId("set-field-api_key")).toHaveAttribute("placeholder", "••••••••");
});

test("non-secret fields blur-save on a configured provider (ollama endpoint)", async ({
  page,
}) => {
  // Owner-hit 2026-07-23 (as the thinking-budget field, since folded into a default):
  // the Test button was the form's only save path — typing into a non-secret field and
  // leaving Settings silently discarded it. Blur now saves.
  await openModels(page);
  await page.getByTestId("set-provider-ollama").click();
  const endpoint = page.getByTestId("set-field-base_url");
  await endpoint.fill("http://127.0.0.1:9999");
  await endpoint.blur();
  await expect(page.getByTestId("set-field-saved-base_url")).toBeVisible();

  // Leave and come back: the value survived (served from the provider's stored values).
  await page.getByTestId("set-back").click();
  await page.getByTestId("set-provider-ollama").click();
  await expect(page.getByTestId("set-field-base_url")).toHaveValue("http://127.0.0.1:9999");
});

// #97: local servers can require a key (oMLX, a proxied `ollama serve`), so the keyless
// provider grew an OPTIONAL secret. It must not turn the form into a keyed one: the endpoint
// stays inline (not behind the Custom endpoint disclosure) and Detect stays usable with the
// key box empty — otherwise every plain-Ollama user is blocked by a field they don't need.
test("an optional key renders below the endpoint without gating Detect (ollama)", async ({
  page,
}) => {
  await openModels(page);
  await page.getByTestId("set-provider-ollama").click();

  await expect(page.getByTestId("set-field-base_url")).toBeVisible();
  await expect(page.getByTestId("set-field-api_key")).toBeVisible();
  await expect(page.getByTestId("set-endpoint-link")).toHaveCount(0);
  await expect(page.getByTestId("set-field-api_key")).toHaveAttribute("type", "password");
  await expect(page.getByTestId("set-test")).toBeEnabled();
  await expect(page.getByTestId("set-test")).toHaveText("Detect");
});

test("an optional key saves, reads back as stored, and can be removed (ollama)", async ({
  page,
}) => {
  await openModels(page);
  page.on("dialog", (d) => d.accept());
  await page.getByTestId("set-provider-ollama").click();

  await page.getByTestId("set-field-api_key").fill("sk-omlx-realkey");
  await page.getByTestId("set-test").click();
  await expect(page.getByTestId("set-provider-ollama")).toBeVisible(); // slid home after saving

  // Reopened: the stored key shows as masked, never as its value, and can be removed.
  await page.getByTestId("set-provider-ollama").click();
  await expect(page.getByTestId("set-field-api_key")).toHaveValue("");
  await expect(page.getByTestId("set-field-api_key")).toHaveAttribute("placeholder", "••••••••");
  await page.getByTestId("set-remove-key").click();

  await page.getByTestId("set-provider-ollama").click();
  await expect(page.getByTestId("set-field-api_key")).toHaveAttribute(
    "placeholder",
    "Only if your server requires one",
  );
  await expect(page.getByTestId("set-remove-key")).toHaveCount(0);
});
