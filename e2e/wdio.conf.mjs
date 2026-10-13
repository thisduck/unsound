import { spawn } from "node:child_process";
import { createConnection, createServer } from "node:net";
import { mkdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { SevereServiceError } from "webdriverio";
import { application, preflight } from "./preflight.mjs";

const children = [];
const root = fileURLToPath(new URL("../", import.meta.url));
const artifacts = fileURLToPath(new URL("artifacts/", import.meta.url));

function start(packageName, env = process.env) {
  const child = spawn(process.execPath, [`${root}node_modules/@crabnebula/${packageName}/cli.js`], {
    cwd: root, env, stdio: "inherit",
  });
  child.on("error", (error) => { child.startError = error; });
  children.push(child);
  return child;
}

async function available(port) {
  await new Promise((resolve, reject) => {
    const server = createServer();
    server.once("error", () => reject(new Error(`Port ${port} is in use; stop the existing test driver first.`)));
    server.listen(port, "127.0.0.1", () => server.close(resolve));
  });
}

async function ready(child, port) {
  const deadline = Date.now() + 30_000;
  while (Date.now() < deadline) {
    if (child.startError) throw child.startError;
    if (child.exitCode !== null || child.signalCode !== null) {
      throw new Error(`Test service on ${port} exited before becoming ready.`);
    }
    const connected = await new Promise((resolve) => {
      const socket = createConnection({ host: "127.0.0.1", port });
      const finish = (ok) => { socket.destroy(); resolve(ok); };
      socket.setTimeout(500);
      socket.once("connect", () => finish(true));
      socket.once("error", () => finish(false));
      socket.once("timeout", () => finish(false));
    });
    if (connected) return;
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw new Error(`Test service on ${port} did not become ready within 30 seconds.`);
}

async function cleanup() {
  await Promise.all(children.splice(0).map(async (child) => {
    if (child.exitCode !== null || child.signalCode !== null) return;
    await new Promise((resolve) => {
      const timer = setTimeout(() => child.kill("SIGKILL"), 2000);
      child.once("exit", () => { clearTimeout(timer); resolve(); });
      child.kill();
    });
  }));
}

process.once("exit", () => { for (const child of children) child.kill("SIGKILL"); });
for (const signal of ["SIGINT", "SIGTERM"]) {
  process.once(signal, async () => { await cleanup(); process.exit(1); });
}

export const config = {
  runner: "local", hostname: "127.0.0.1", port: 4444,
  specs: [fileURLToPath(new URL("specs/*.spec.mjs", import.meta.url))],
  maxInstances: 1,
  capabilities: [{ "tauri:options": { application } }],
  framework: "mocha", reporters: ["spec"],
  mochaOpts: { timeout: 60_000 },
  waitforTimeout: 10_000, connectionRetryTimeout: 30_000, connectionRetryCount: 0,
  logLevel: "warn",
  onPrepare: async () => {
    try {
      preflight();
      await available(3000);
      await available(4444);
      // The repository secret has a purpose-specific name; CrabNebula's
      // installed backend reads CN_API_KEY.
      const env = { ...process.env, CN_API_KEY: process.env.CN_API_KEY_WEBDRIVER };
      await ready(start("test-runner-backend", env), 3000);
      await ready(start("tauri-driver", {
        ...env, REMOTE_WEBDRIVER_URL: "http://127.0.0.1:3000",
      }), 4444);
    } catch (error) {
      await cleanup();
      // Ordinary hook errors are logged and swallowed by WDIO. This type
      // aborts before any worker can connect to a missing or unrelated driver.
      throw new SevereServiceError(error.message);
    }
  },
  afterTest: async function (_test, _context, { passed }) {
    if (!passed) {
      await mkdir(artifacts, { recursive: true });
      const { browser } = await import("@wdio/globals");
      await browser.saveScreenshot(`${artifacts}/failure-${Date.now()}.png`);
    }
  },
  onComplete: cleanup,
};
