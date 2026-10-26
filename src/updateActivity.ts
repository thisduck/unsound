// Work and unsaved notes must settle before an update can restart the app.
let pending = 0;
const unsaved = new Set<string>();
export async function trackWork<T>(work: () => Promise<T>): Promise<T> {
  pending++;
  try { return await work(); } finally { pending--; }
}
export function markUnsaved(id: string, dirty: boolean) {
  if (dirty) unsaved.add(id); else unsaved.delete(id);
}
export function updateWorkBlocker(): string | null {
  if (unsaved.size) return "Save your meeting notes before restarting to update.";
  return pending ? "Wait for current work to finish before restarting to update." : null;
}
