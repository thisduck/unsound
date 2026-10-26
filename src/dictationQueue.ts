export interface DictationState<T> {
  active: T | null;
  capturing: boolean;
  pending: number;
}

interface Ports<T, R> {
  start(take: T): Promise<void>;
  finish(take: T, cancelled: boolean): Promise<R | null>;
  process(take: T, recording: R): Promise<void>;
  changed(state: DictationState<T>): void;
  failed(take: T, error: unknown): void;
}

// Mic operations run in input order, independently of the processing queue.
// Even down/up/down during device startup captures two distinct takes. The
// processing queue includes delivery, so text always arrives in spoken order.
export function createDictationQueue<T, R>(ports: Ports<T, R>) {
  let active: { take: T; started: boolean } | null = null;
  let captures = 0;
  let pending = 0;
  let captureTail = Promise.resolve();
  let processTail = Promise.resolve();
  const state = (): DictationState<T> => ({ active: active?.take ?? null, capturing: captures > 0 || active !== null, pending });
  const changed = () => ports.changed(state());
  const capture = (work: () => Promise<void>) => {
    captures++;
    changed();
    captureTail = captureTail.then(work).finally(() => {
      captures--;
      changed();
    });
  };

  const start = (take: T) => {
    if (active) return;
    const session = { take, started: false };
    active = session;
    capture(async () => {
      try {
        await ports.start(take);
        session.started = true;
      } catch (error) {
        if (active === session) active = null;
        ports.failed(take, error);
      }
    });
  };

  const stop = (cancelled = false) => {
    const session = active;
    if (!session) return;
    active = null;
    if (!cancelled) pending++;
    capture(async () => {
      let queued = false;
      try {
        if (!session.started) return;
        const recording = await ports.finish(session.take, cancelled);
        if (cancelled || recording === null) return;
        queued = true;
        processTail = processTail.then(async () => {
          try {
            await ports.process(session.take, recording);
          } catch (error) {
            ports.failed(session.take, error);
          } finally {
            pending--;
            changed();
          }
        });
      } catch (error) {
        ports.failed(session.take, error);
      } finally {
        if (!cancelled && !queued) pending--;
      }
    });
  };

  return { start, stop, state };
}
