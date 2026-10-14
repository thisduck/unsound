import { spawn, execFile } from "node:child_process";
import { promisify } from "node:util";
import { randomUUID } from "node:crypto";
import { mkdir, writeFile, access } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";
import os from "node:os";
const exec = promisify(execFile);
const args = process.argv.slice(2);
const option = (key, fallback) => { const i = args.indexOf(`--${key}`); return i < 0 ? fallback : args[i+1]; };
const model = option("model"), audio = option("audio");
if (!model || !audio) throw new Error("Usage: npm run bench:meeting -- --model /path/ggml-model.bin --audio /path/speech.wav [--minutes 60] [--speed 20] [--restart] [--report /path/report.json]");
await access(model); await access(audio);
const minutes = Number(option("minutes", "60")), speed = Number(option("speed", "20"));
if (!Number.isInteger(minutes) || minutes < 1 || minutes > 90 || !Number.isInteger(speed) || speed < 1 || speed > 20) throw new Error("minutes must be 1–90; speed must be 1–20");
const restart = args.includes("--restart");
const binary = fileURLToPath(new URL("bin/unsound-e2e", import.meta.url));
const reportPath = path.resolve(option("report", "e2e/artifacts/meeting-benchmark.json"));
const id = `e2e-${randomUUID()}`;
const env = { ...process.env, UNSOUND_BENCH_MODEL: path.resolve(model), UNSOUND_BENCH_AUDIO: path.resolve(audio), UNSOUND_BENCH_MINUTES: String(minutes), UNSOUND_BENCH_SPEED: String(speed), UNSOUND_BENCH_RESTART: restart ? "1" : "0" };
async function run(scenario) {
  const child = spawn(binary, ["--meeting-fixture-scenario", scenario, id], { env, stdio: ["ignore", "pipe", "pipe"] });
  let logs = "", stopped, report, killed = false, sampling = false;
  const samples = [], start = performance.now();
  const consume = (chunk) => {
    logs = (logs + chunk).slice(-4_000_000);
    const stop = logs.match(/MEETING_BENCH_STOP (.*)\n/); if (stop) stopped = JSON.parse(stop[1]);
    const result = logs.match(/MEETING_BENCH_REPORT (.*)\n/); if (result) report = JSON.parse(result[1]);
    if (scenario === "soak" && restart && !killed && logs.includes("MEETING_BENCH_RESTART_READY")) { killed = true; child.kill("SIGKILL"); }
  };
  child.stdout.on("data", consume); child.stderr.on("data", consume);
  const sampler = setInterval(async () => {
    if (sampling || !child.pid) return; sampling = true;
    try { const { stdout } = await exec("ps", ["-o", "rss=,%cpu=", "-p", String(child.pid)]); const [rss,cpu] = stdout.trim().split(/\s+/).map(Number); if (Number.isFinite(rss) && Number.isFinite(cpu)) samples.push({ seconds:(performance.now()-start)/1000,rssBytes:rss*1024,cpuPercent:cpu }); }
    catch {} finally { sampling = false; }
  }, 1000);
  const timeout = setTimeout(() => child.kill("SIGKILL"), (minutes*60/speed + 1200)*1000);
  try {
    const { code, signal } = await new Promise((resolve,reject) => {child.once("error",reject);child.once("close",(code,signal)=>resolve({code,signal}));});
    if (!(killed && signal === "SIGKILL") && !(code === 0 && report)) throw new Error(`Benchmark failed (${code ?? signal}): ${logs.slice(-5000)}`);
    const warm = samples.find((s) => s.seconds >= 30);
    return { report, stopped, peakRssBytes:samples.length ? Math.max(...samples.map(s=>s.rssBytes)) : null,rssGrowthAfter30sBytes:warm && samples.length ? samples.at(-1).rssBytes-warm.rssBytes : null,samples };
  } finally {clearInterval(sampler);clearTimeout(timeout);if(child.exitCode===null && child.signalCode===null) child.kill("SIGKILL");await mkdir(path.dirname(reportPath),{recursive:true});await writeFile(`${reportPath}.${scenario}.log`,logs);}
}
const recording = await run("soak");
const recovery = restart ? await run("soak-recover") : null;
await writeFile(reportPath,JSON.stringify({ recordedAt:new Date().toISOString(),host:{platform:os.platform(),release:os.release(),arch:os.arch(),cpu:os.cpus()[0]?.model,logicalCpus:os.cpus().length,memoryBytes:os.totalmem()},workload:{minutesPerChannel:minutes,channels:2,speed,restart},mode:speed===1?"wall-clock":"accelerated",model:path.basename(model),recording,recovery },null,2)+"\n");
console.log(`Benchmark report: ${reportPath}`);
console.log(JSON.stringify({recording:recording.report?.metrics ?? recording.stopped,recovery:recovery?.report?.metrics,peakRssBytes:Math.max(recording.peakRssBytes ?? 0,recovery?.peakRssBytes ?? 0)},null,2));
