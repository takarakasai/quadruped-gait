//! Zenoh subscriber for the live-visualization stream: receives
//! [`GaitVizFrame`](crate::viz::GaitVizFrame)s published by a gait runner
//! (`go2-gait-runner --viz`) and keeps the most recent one for a viewer to
//! poll. Pure transport — how frames are applied to a model is the
//! viewer's concern.
//!
//! Feature-gated behind `viz-sub` so wire-type-only consumers (publishers,
//! log tooling) don't pull zenoh.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use zenoh::Wait;

use crate::viz::GaitVizFrame;

/// Background Zenoh subscriber holding the latest received frame.
///
/// Dropping the subscriber signals the background thread to exit (it polls
/// with a 200 ms timeout, so shutdown is prompt but not instant).
pub struct VizSubscriber {
    latest: Arc<Mutex<Option<GaitVizFrame>>>,
    running: Arc<AtomicBool>,
    _handle: std::thread::JoinHandle<()>,
}

impl VizSubscriber {
    /// `endpoint = Some(ep)` connects to a Zenoh peer listening at `ep` (TCP)
    /// and disables multicast — use it when multicast discovery isn't
    /// available (same host / WSL2). `None` = auto multicast discovery.
    pub fn new(key: &str, endpoint: Option<&str>) -> Result<Self, String> {
        let latest: Arc<Mutex<Option<GaitVizFrame>>> = Arc::new(Mutex::new(None));
        let running = Arc::new(AtomicBool::new(true));
        let l2 = latest.clone();
        let r2 = running.clone();
        let key = key.to_string();
        let mut config = zenoh::Config::default();
        if let Some(ep) = endpoint {
            config
                .insert_json5("connect/endpoints", &format!("[\"{ep}\"]"))
                .map_err(|e| format!("zenoh connect endpoint '{ep}': {e}"))?;
            let _ = config.insert_json5("scouting/multicast/enabled", "false");
        }
        let handle = std::thread::Builder::new()
            .name("viz-sub".into())
            .spawn(move || {
                let session = match zenoh::open(config).wait() {
                    Ok(s) => s,
                    Err(e) => {
                        eprintln!("viz-sub: zenoh open failed: {e}");
                        return;
                    }
                };
                let sub = match session.declare_subscriber(&key).wait() {
                    Ok(s) => s,
                    Err(e) => {
                        eprintln!("viz-sub: subscribe '{key}' failed: {e}");
                        return;
                    }
                };
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
        Ok(Self {
            latest,
            running,
            _handle: handle,
        })
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
