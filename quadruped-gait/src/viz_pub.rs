//! Zenoh publisher for the live-visualization stream: sends
//! [`GaitVizFrame`](crate::viz::GaitVizFrame)s a viewer (e.g. the `articara`
//! GUI) picks up with [`VizSubscriber`](crate::viz_sub::VizSubscriber).
//!
//! The counterpart to [`crate::viz_sub`], and the reason it exists: publishing
//! correctly is more than a `put`. The stream pairing has to hold, the frames
//! must not be built or sent on the control loop, and back-pressure has to drop
//! rather than block. Every runner that rolled its own got to rediscover that.
//! See `doc/viz_publisher.md` for the contract this implements.
//!
//! What stays with the caller is the robot-specific part: building the frames
//! (joint order, sign conventions, where a measured pose comes from). This
//! module owns the transport and the invariants that are the same everywhere.
//!
//! Feature-gated behind `viz-pub` so wire-type-only consumers don't pull zenoh.

use std::sync::mpsc::{sync_channel, SyncSender};

use zenoh::Wait;

use crate::viz::{GaitVizFrame, VIZ_KEY_MEASURED, VIZ_KEY_PLANNED};
use crate::viz_net::VizEndpoints;

/// Frames queued for the publisher thread. Bounded, so a stalled network can't
/// grow it without limit; a full queue drops rather than blocking, which is the
/// right trade on a lossy latest-wins channel.
const QUEUE_DEPTH: usize = 8;

/// Publisher setup: where to publish, how often, and how to reach the network.
#[derive(Clone, Debug)]
pub struct VizPublisherConfig {
    /// Key for the commanded stream.
    pub key_planned: String,
    /// Key for the measured stream. Empty = don't publish one. Must differ from
    /// [`Self::key_planned`]; see [`VizPublisher::new`].
    pub key_measured: String,
    /// Target publish rate. The publisher divides it into the caller's tick
    /// period, so [`VizPublisher::publish`] can be called every control tick.
    pub rate_hz: f64,
    /// The caller's tick period, seconds.
    pub dt: f64,
    /// How the session joins the network.
    pub endpoints: VizEndpoints,
}

impl Default for VizPublisherConfig {
    fn default() -> Self {
        Self {
            key_planned: VIZ_KEY_PLANNED.to_string(),
            key_measured: VIZ_KEY_MEASURED.to_string(),
            rate_hz: 100.0,
            dt: 0.002,
            endpoints: VizEndpoints::auto(),
        }
    }
}

impl VizPublisherConfig {
    /// Re-namespace both keys under `robot`, replacing the leading chunk
    /// (`<robot>/gait/<stream>`; see `doc/viz_publisher.md` §5.4).
    pub fn with_robot(mut self, robot: &str) -> Self {
        fn swap(key: &str, robot: &str) -> String {
            match key.split_once('/') {
                Some((_, rest)) => format!("{robot}/{rest}"),
                None => robot.to_string(),
            }
        }
        self.key_planned = swap(&self.key_planned, robot);
        if !self.key_measured.is_empty() {
            self.key_measured = swap(&self.key_measured, robot);
        }
        self
    }
}

/// Background Zenoh publisher for the viz stream.
///
/// [`Self::publish`] is safe to call from a control loop: it builds nothing on
/// the ticks it skips, and hands the frames it does build to a thread that owns
/// the session, so no serialization, syscall or network write happens on the
/// caller's thread.
///
/// Dropping the publisher closes the channel, which ends the thread.
pub struct VizPublisher {
    tx: SyncSender<(String, GaitVizFrame)>,
    key_planned: String,
    key_measured: String,
    seq: u64,
    period: u32,
    since: u32,
    dropped: u64,
}

impl VizPublisher {
    /// Open the session and start the publisher thread.
    ///
    /// A measured key equal to the planned one is refused — not as an error,
    /// but by dropping the measured stream with a warning. The channel is
    /// latest-wins, so one key carrying both poses would have them overwrite
    /// each other and a viewer would flip between command and response, which
    /// is worse than not having the second stream at all.
    pub fn new(cfg: VizPublisherConfig) -> Result<Self, String> {
        let mut config = zenoh::Config::default();
        cfg.endpoints.apply(&mut config)?;
        let session = zenoh::open(config)
            .wait()
            .map_err(|e| format!("zenoh open: {e}"))?;

        let key_measured = if cfg.key_measured == cfg.key_planned {
            eprintln!(
                "viz-pub: measured key '{}' is the same stream as the planned one — \
                 measured not published (the two must be distinct keys)",
                cfg.key_measured
            );
            String::new()
        } else {
            cfg.key_measured
        };

        let (tx, rx) = sync_channel::<(String, GaitVizFrame)>(QUEUE_DEPTH);
        std::thread::Builder::new()
            .name("viz-pub".into())
            .spawn(move || {
                // Ends when the sender is dropped (publisher torn down).
                for (key, frame) in rx {
                    if let Ok(json) = serde_json::to_vec(&frame) {
                        let _ = session
                            .put(&key, json)
                            .encoding(zenoh::bytes::Encoding::APPLICATION_JSON)
                            .wait();
                    }
                }
            })
            .map_err(|e| format!("spawn viz-pub thread: {e}"))?;

        Ok(Self {
            tx,
            key_planned: cfg.key_planned,
            key_measured,
            seq: 0,
            period: ((1.0 / cfg.rate_hz.max(1.0)) / cfg.dt).round().max(1.0) as u32,
            since: 0,
            dropped: 0,
        })
    }

    /// Key of the commanded stream.
    pub fn key_planned(&self) -> &str {
        &self.key_planned
    }

    /// Key of the measured stream, or `None` when it isn't being published.
    pub fn key_measured(&self) -> Option<&str> {
        (!self.key_measured.is_empty()).then_some(self.key_measured.as_str())
    }

    /// Frames dropped because the publisher thread couldn't keep up. Worth
    /// reporting on shutdown: a throttled stream otherwise looks healthy.
    pub fn dropped(&self) -> u64 {
        self.dropped
    }

    /// Call every control tick. `build` runs **only** on the ticks that
    /// actually publish, so frame construction costs nothing in between; it
    /// returns the commanded frame and, if the robot reports state back, the
    /// matching measured one.
    ///
    /// Both frames are stamped with the same sequence number here, overwriting
    /// whatever `build` put in [`GaitVizFrame::seq`]. That pairing is what
    /// bounds the skew a viewer sees between the two streams to one publish
    /// period, and it is too easy to get wrong by hand to leave to the caller.
    pub fn publish<F>(&mut self, build: F)
    where
        F: FnOnce() -> (GaitVizFrame, Option<GaitVizFrame>),
    {
        self.since += 1;
        if self.since < self.period {
            return;
        }
        self.since = 0;

        let (mut planned, measured) = build();
        planned.seq = self.seq;
        if let (Some(mut m), Some(key)) = (measured, self.key_measured().map(str::to_string)) {
            m.seq = self.seq;
            self.hand_off(key, m);
        }
        self.seq += 1;
        let key = self.key_planned.clone();
        self.hand_off(key, planned);
    }

    /// Queue one frame, dropping it if the queue is full. Never blocks: a slow
    /// or stalled network costs frames, not control-loop deadlines.
    fn hand_off(&mut self, key: String, frame: GaitVizFrame) {
        if self.tx.try_send((key, frame)).is_err() {
            self.dropped += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(seq: u64) -> GaitVizFrame {
        GaitVizFrame {
            version: crate::viz::VIZ_FORMAT_VERSION,
            seq,
            t_s: 0.0,
            pose: [0.0; 4],
            pose_rp: [0.0; 2],
            joints: [0.0; 12],
            stance: [true; 4],
        }
    }

    fn publisher(cfg: VizPublisherConfig) -> VizPublisher {
        VizPublisher::new(VizPublisherConfig {
            // Nothing to discover and nothing to reach: the session opens, the
            // thread runs, and the puts go nowhere — which is all these tests
            // need, without touching the network.
            endpoints: VizEndpoints::auto().with_multicast(false),
            ..cfg
        })
        .expect("open publisher")
    }

    #[test]
    fn the_robot_namespace_moves_both_keys_together() {
        let cfg = VizPublisherConfig::default().with_robot("go2-01");
        assert_eq!(cfg.key_planned, "go2-01/gait/planned");
        assert_eq!(cfg.key_measured, "go2-01/gait/measured");
    }

    /// The rate divides into the caller's tick period: at 100 Hz on a 500 Hz
    /// loop, one tick in five builds a frame and the other four cost nothing.
    #[test]
    fn only_the_publishing_ticks_build_a_frame() {
        let mut p = publisher(VizPublisherConfig {
            rate_hz: 100.0,
            dt: 0.002,
            ..Default::default()
        });
        let mut built = 0;
        for _ in 0..10 {
            p.publish(|| {
                built += 1;
                (frame(999), None)
            });
        }
        assert_eq!(built, 2, "10 ticks at 5:1 downsampling");
    }

    /// The pairing the viewer relies on: whatever the caller stamped, the two
    /// frames of one tick go out under the same sequence number, and it
    /// advances once per publish rather than once per frame.
    #[test]
    fn both_streams_of_a_tick_share_one_sequence_number() {
        let mut p = publisher(VizPublisherConfig {
            rate_hz: 500.0,
            dt: 0.002,
            ..Default::default()
        });
        let mut seen: Vec<(u64, u64)> = Vec::new();
        for _ in 0..3 {
            p.publish(|| {
                let (a, b) = (frame(42), frame(7)); // deliberately mismatched
                seen.push((a.seq, b.seq));
                (a, Some(b))
            });
        }
        assert_eq!(seen.len(), 3);
        assert_eq!(p.seq, 3, "one sequence number per publish, not per frame");
    }

    /// One key can't carry two poses on a latest-wins channel, so the measured
    /// stream is dropped rather than corrupting the commanded one.
    #[test]
    fn a_measured_key_equal_to_the_planned_one_is_refused() {
        let p = publisher(VizPublisherConfig {
            key_planned: "go2/gait/planned".into(),
            key_measured: "go2/gait/planned".into(),
            ..Default::default()
        });
        assert_eq!(p.key_planned(), "go2/gait/planned");
        assert_eq!(p.key_measured(), None);
    }

    #[test]
    fn an_empty_measured_key_publishes_the_planned_stream_alone() {
        let p = publisher(VizPublisherConfig {
            key_measured: String::new(),
            ..Default::default()
        });
        assert_eq!(p.key_measured(), None);
    }
}
