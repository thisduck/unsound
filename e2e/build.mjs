import { spawnSync } from "node:child_process";
import { copyFileSync, mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../", import.meta.url));
if (process.platform !== "darwin") throw new Error("The native harness currently supports macOS.");
const result = spawnSync("npm", ["run", "tauri", "build", "--", "--debug", "--no-bundle",
  "--features", "automation", "--config", "src-tauri/tauri.e2e.conf.json"], {
  cwd: root, stdio: "inherit",
  // Keep the output path predictable even in a shell with a custom Cargo target.
  env: { ...process.env, CARGO_TARGET_DIR: `${root}src-tauri/target` },
});
if (result.error) throw result.error;
if (result.status !== 0) process.exit(result.status ?? 1);
mkdirSync(new URL("bin/", import.meta.url), { recursive: true });
copyFileSync(new URL("../src-tauri/target/debug/unsound", import.meta.url),
  new URL("bin/unsound-e2e", import.meta.url));
