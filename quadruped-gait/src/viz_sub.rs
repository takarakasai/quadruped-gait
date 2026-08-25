//! Zenoh subscriber for the live-visualization stream: receives
//! [`GaitVizFrame`](crate::viz::GaitVizFrame)s published by a gait runner
//! (`go2-gait-runner --viz`) and keeps the most recent one for a viewer to
//! poll. Pure transport — how frames are applied to a model is the
//! viewer's concern.
//!
//! Feature-gated behind `viz-sub` so wire-type-only consumers (publishers,
//! log tooling) don't pull zenoh. The publishing side is [`crate::viz_pub`];
//! both take the same [`VizEndpoints`] so either end can listen, connect, or
//! discover.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use zenoh::Wait;

use crate::viz::GaitVizFrame;
use crate::viz_net::VizEndpoints;

/// A Zenoh session viz subscribers ride on.
///
/// One session per *network configuration*, not per stream. Two sessions that
/// both listen on the same endpoint collide outright — the second one fails to
/// bind — and even where they don't, a second session to the same publisher is
/// pure overhead. A viewer watching a planned and a measured stream from one
/// runner should open this once and [`Self::subscribe`] twice.
///
/// Cloning is cheap (Zenoh sessions are reference-counted); the session closes
/// when the last handle, including any live subscriber's, goes away.
#[derive(Clone)]
pub struct VizSession {
    session: zenoh::Session,
}

impl VizSession {
    /// Open a session, joining the network as `endpoints` says.
    ///
    /// Failures surface here rather than in a background thread: a viewer that
    /// reports itself subscribed while its session never opened is worse than
    /// one that says it couldn't start.
    pub fn open(endpoints: &VizEndpoints) -> Result<Self, String> {
        let mut config = zenoh::Config::default();
        endpoints.apply(&mut config)?;
        let session = zenoh::open(config)
            .wait()
            .map_err(|e| format!("zenoh open ({}): {e}", endpoints.describe()))?;
        Ok(Self { session })
    }

    /// Subscribe to `key` and start polling it in the background.
    pub fn subscribe(&self, key: &str) -> Result<VizSubscriber, String> {
        let latest: Arc<Mutex<Option<GaitVizFrame>>> = Arc::new(Mutex::new(None));
        let running = Arc::new(AtomicBool::new(true));
        let l2 = latest.clone();
        let r2 = running.clone();
        let sub = self
            .session
            .declare_subscriber(key)
            .wait()
            .map_err(|e| format!("zenoh subscribe '{key}': {e}"))?;
        let handle = std::thread::Builder::new()
            .name("viz-sub".into())
            .spawn(move || {
                // recv_timeout (not blocking recv) so the thread can notice
                // the stop flag and exit when the subscriber is dropped.
                while r2.load(Ordering::Relaxed) {
                    match sub.recv_timeout(Duration::from_millis(200)) {
                        Ok(Some(sample)) => {
                            let bytes = sample.payload().to_bytes();
                            if let Ok(frame) = serde_json::from_slice::<GaitVizFrame>(&bytes) {
                                if frame.is_compatible() {
                                    *l2.lock().unwrap() = Some(frame);
                                }
                            }
                        }
                        Ok(None) => {} // timeout — re-check the stop flag
                        Err(_) => break,
                    }
                }
            })
            .map_err(|e| format!("spawn viz-sub thread: {e}"))?;
        Ok(VizSubscriber {
            latest,
            running,
            _session: self.session.clone(),
            _handle: handle,
        })
    }
}

/// Background Zenoh subscriber holding the latest received frame.
///
/// Dropping the subscriber signals the background thread to exit (it polls
/// with a 200 ms timeout, so shutdown is prompt but not instant).
pub struct VizSubscriber {
    latest: Arc<Mutex<Option<GaitVizFrame>>>,
    running: Arc<AtomicBool>,
    /// Keeps the session alive for as long as this subscriber polls it.
    _session: zenoh::Session,
    _handle: std::thread::JoinHandle<()>,
}

impl VizSubscriber {
    /// Open a session of its own and subscribe to `key` — the single-stream
    /// convenience. A viewer taking two streams off one publisher should open a
    /// [`VizSession`] and subscribe twice instead, so they share it.
    ///
    /// A viewer usually connects to the publisher
    /// ([`VizEndpoints::connect`]), but the reverse works too: listen and let
    /// the robot dial in, which is what you want when the robot's address is
    /// the one that moves. [`VizEndpoints::auto`] uses multicast discovery.
    pub fn new(key: &str, endpoints: &VizEndpoints) -> Result<Self, String> {
        VizSession::open(endpoints)?.subscribe(key)
    }

    /// Take (consume) the latest frame, if a new one has arrived since the
    /// previous call.
    pub fn take_latest(&self) -> Option<GaitVizFrame> {
        self.latest.lock().ok().and_then(|mut g| g.take())
    }
}

impl Drop for VizSubscriber {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
    }
}
