// Executes the real native backend without WebDriver or cloud credentials.
import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import { mkdir, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

const application = fileURLToPath(new URL("bin/unsound-e2e", import.meta.url));
const artifacts = new URL("artifacts/", import.meta.url);
const id = `e2e-${randomUUID()}`;
async function scenario(name, fixtureId = id) {
  const child = spawn(application, ["--meeting-fixture-scenario", name, fixtureId], { stdio: ["ignore", "pipe", "pipe"] });
  let output = "", killed = false;
  const timer = setTimeout(() => child.kill("SIGKILL"), 60_000);
  const collect = (chunk) => {
    output = (output + chunk).slice(-2_000_000);
    if (name === "crash-write" && !killed && output.includes("MEETING_FIXTURE_KILL_READY")) {
      killed = true;
      child.kill("SIGKILL");
    }
  };
  child.stdout.on("data", collect);
  child.stderr.on("data", collect);
  try {
    const result = await new Promise((resolve, reject) => {
      child.once("error", reject);
      child.once("close", (code, signal) => resolve({ code, signal }));
    });
    await mkdir(artifacts, { recursive: true });
    await writeFile(new URL(`backend-${name}.log`, artifacts), output);
    const passed = name === "crash-write"
      ? killed && result.signal === "SIGKILL"
      : result.code === 0 && output.includes(`MEETING_FIXTURE_PASS ${name}`);
    if (!passed) throw new Error(`${name} failed (${result.code ?? result.signal}):\n${output}`);
    console.log(`PASS native backend: ${name}`);
  } finally { clearTimeout(timer); if (child.exitCode === null && child.signalCode === null) child.kill("SIGKILL"); }
}
await scenario("recording-indicator");
await scenario("failures");
await scenario("crash-write", `${id}-crash`);
await scenario("crash-read", `${id}-crash`);
await scenario("crash-cleanup", `${id}-crash`);
