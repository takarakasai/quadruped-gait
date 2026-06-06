//! Background solve worker for the MPC gait controllers.
//!
//! The MPC QP solve is expensive (full-centroidal ≈ 0.4 s per solve) and
//! was previously run **synchronously inside `tick()`**. Because the
//! `articara` GUI drives `tick()` from the eframe update loop on the UI
//! thread, a slow solve blocked the whole window — combined with the
//! per-frame `request_repaint()`, once one solve exceeded the `dt_per_step`
//! re-solve window the controller re-solved *every* frame and the app
//! appeared frozen ("固まる"). See `memory/project_mpc_frame_bug.md` /
//! the freeze analysis.
//!
//! [`AsyncJobWorker`] moves the solve onto a dedicated background thread.
//! The control loop submits a self-contained solve closure when a new
//! solution is due, keeps using the previous solution until the fresh one
//! lands (zero-order-hold — the standard async-MPC pattern), and never
//! blocks. A slow QP now only delays the *next* solution; it can no longer
//! stall the caller.
//!
//! The worker is generic over the solution type `O`. The caller builds a
//! `FnOnce() -> O + Send + 'static` job that owns a clone of its solver
//! plus the per-solve inputs, so the worker needs to know nothing about
//! the MPC formulation.

use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::thread::{self, JoinHandle};

type Job<O> = (u64, Box<dyn FnOnce() -> O + Send + 'static>);

/// A single long-lived worker thread that runs solve jobs off the caller's
/// thread. At most one job is in flight at a time (enforced by the caller
/// via [`Self::is_busy`]); results are delivered back over a channel and
/// drained non-blockingly with [`Self::poll`].
pub struct AsyncJobWorker<O: Send + 'static> {
    job_tx: Sender<Job<O>>,
    res_rx: Receiver<(u64, O)>,
    /// True between [`Self::submit`] and the matching [`Self::poll`] that
    /// drains its result. Used by the caller to avoid queueing a second
    /// solve before the first finishes.
    busy: bool,
    /// Monotonic tag stamped onto each submitted job. [`Self::reset`] bumps
    /// it so results from solves submitted *before* a reset are recognised
    /// as stale and discarded (rather than repopulating a just-cleared
    /// solution).
    epoch: u64,
    _worker: JoinHandle<()>,
}

impl<O: Send + 'static> AsyncJobWorker<O> {
    pub fn new() -> Self {
        let (job_tx, job_rx) = mpsc::channel::<Job<O>>();
        let (res_tx, res_rx) = mpsc::channel::<(u64, O)>();
        let worker = thread::Builder::new()
            .name("mpc-solver".to_string())
            .spawn(move || {
                // Exit cleanly when the caller (and thus `job_tx`) is
                // dropped: `recv()` then returns `Err`.
                while let Ok((epoch, job)) = job_rx.recv() {
                    let out = job();
                    if res_tx.send((epoch, out)).is_err() {
                        break;
                    }
                }
            })
            .expect("spawn mpc-solver thread");
        Self {
            job_tx,
            res_rx,
            busy: false,
            epoch: 0,
            _worker: worker,
        }
    }

    /// Whether a solve is currently in flight. Callers should only
    /// [`Self::submit`] when this is `false`.
    pub fn is_busy(&self) -> bool {
        self.busy
    }

    /// Hand a solve job to the worker thread. Marks the worker busy until
    /// the result is drained by [`Self::poll`]. If the worker thread has
    /// died the send fails silently and `busy` stays `false`.
    pub fn submit(&mut self, job: impl FnOnce() -> O + Send + 'static) {
        if self.job_tx.send((self.epoch, Box::new(job))).is_ok() {
            self.busy = true;
        }
    }

    /// Non-blocking: return the freshest finished result, or `None` if the
    /// worker hasn't produced one since the last poll. Results tagged with
    /// a stale epoch (submitted before a [`Self::reset`]) are discarded.
    /// Clears `busy` once the in-flight solve has reported back.
    pub fn poll(&mut self) -> Option<O> {
        let mut latest = None;
        loop {
            match self.res_rx.try_recv() {
                Ok((epoch, out)) => {
                    // One result per in-flight solve, so any arrival frees
                    // the worker — even a stale (post-reset) one.
                    self.busy = false;
                    if epoch == self.epoch {
                        latest = Some(out);
                    }
                }
                Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => break,
            }
        }
        latest
    }

    /// Invalidate any in-flight solve so its result won't repopulate a
    /// just-cleared solution. Used by controller `reset()`. The worker
    /// thread keeps running; the orphaned result is dropped on the next
    /// [`Self::poll`].
    pub fn reset(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
    }
}

impl<O: Send + 'static> Default for AsyncJobWorker<O> {
    fn default() -> Self {
        Self::new()
    }
}

// A worker owns a thread + channels, which aren't themselves cloneable.
// The controllers that embed one still want to derive `Clone`/`Debug`, so
// we provide both: cloning yields a *fresh independent worker* (its own
// thread, no in-flight solve carried over), and `Debug` skips the
// non-printable internals.
impl<O: Send + 'static> Clone for AsyncJobWorker<O> {
    fn clone(&self) -> Self {
        Self::new()
    }
}

impl<O: Send + 'static> std::fmt::Debug for AsyncJobWorker<O> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AsyncJobWorker")
            .field("busy", &self.busy)
            .field("epoch", &self.epoch)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runs_job_off_thread_and_polls_result() {
        let mut w = AsyncJobWorker::<i32>::new();
        assert!(!w.is_busy());
        w.submit(|| 21 * 2);
        assert!(w.is_busy());
        // Spin until the worker reports back.
        let mut got = None;
        for _ in 0..100_000 {
            if let Some(v) = w.poll() {
                got = Some(v);
                break;
            }
            std::thread::yield_now();
        }
        assert_eq!(got, Some(42));
        assert!(!w.is_busy());
    }

    #[test]
    fn reset_discards_in_flight_result() {
        let mut w = AsyncJobWorker::<i32>::new();
        w.submit(|| 7);
        w.reset(); // bump epoch: the in-flight 7 is now stale
        // Drain until busy clears; the stale result must not surface.
        let mut surfaced = None;
        for _ in 0..100_000 {
            if let Some(v) = w.poll() {
                surfaced = Some(v);
            }
            if !w.is_busy() {
                break;
            }
            std::thread::yield_now();
        }
        assert_eq!(surfaced, None);
        assert!(!w.is_busy());
    }
}
