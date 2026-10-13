import { accessSync, constants } from "node:fs";
import { fileURLToPath } from "node:url";

export const application = fileURLToPath(new URL("bin/unsound-e2e", import.meta.url));

export function preflight() {
  if (process.platform !== "darwin") throw new Error("The native harness currently supports macOS.");
  if (!process.env.CN_API_KEY_WEBDRIVER) {
    throw new Error("Native UI tests require CN_API_KEY_WEBDRIVER for CrabNebula on macOS. No tests ran.");
  }
  try { accessSync(application, constants.X_OK); }
  catch { throw new Error("The isolated E2E app is missing. Run npm run e2e:build first."); }
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  try { preflight(); console.log("Native UI test prerequisites are present."); }
  catch (error) { console.error(error.message); process.exitCode = 1; }
}
