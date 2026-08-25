//! Zenoh session shape shared by the viz publisher and subscriber: **how** a
//! node joins the network, independent of which direction the frames flow.
//!
//! Zenoh peers are symmetric — either end can listen, connect, or discover the
//! other by multicast — but an application that hard-codes one of those forces
//! a topology on its deployment. The viz stream ran that way for a while
//! (publisher listens, viewer connects), which works on a LAN and falls over
//! the moment the robot's address is the one that moves, or a router sits in
//! between. [`VizEndpoints`] carries the choice instead of assuming it.
//!
//! See `doc/viz_publisher.md` §5.2 for the deployment topologies these cover.

/// How a viz session joins the Zenoh network.
///
/// The default is multicast discovery, which needs no addresses at all and is
/// the right answer on a LAN that carries multicast. `listen` and `connect` are
/// not exclusive — a node may accept peers *and* dial out, e.g. listen for
/// viewers while connecting to a router.
///
/// # Multicast
/// Setting any endpoint turns multicast scouting **off** by default, because
/// the usual reason to name an address is that discovery doesn't work here.
/// [`Self::with_multicast`] overrides that either way.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VizEndpoints {
    /// Endpoints to listen on, e.g. `tcp/0.0.0.0:7447`. Peers connect in.
    pub listen: Vec<String>,
    /// Endpoints to connect to, e.g. `tcp/192.168.123.161:7447`.
    pub connect: Vec<String>,
    /// Multicast scouting. `None` = the default rule above.
    pub multicast: Option<bool>,
}

impl VizEndpoints {
    /// Multicast discovery, no addresses. The LAN default.
    pub fn auto() -> Self {
        Self::default()
    }

    /// Listen on `eps` and wait for the other end to connect.
    pub fn listen<I, S>(eps: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            listen: eps.into_iter().map(Into::into).collect(),
            ..Self::default()
        }
    }

    /// Connect out to `eps` — the other end, or a router between them.
    pub fn connect<I, S>(eps: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            connect: eps.into_iter().map(Into::into).collect(),
            ..Self::default()
        }
    }

    /// Force multicast scouting on or off, overriding the default rule.
    pub fn with_multicast(mut self, on: bool) -> Self {
        self.multicast = Some(on);
        self
    }

    /// Add listen endpoints to an existing configuration.
    pub fn also_listen<I, S>(mut self, eps: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.listen.extend(eps.into_iter().map(Into::into));
        self
    }

    /// Add connect endpoints to an existing configuration.
    pub fn also_connect<I, S>(mut self, eps: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.connect.extend(eps.into_iter().map(Into::into));
        self
    }

    /// Whether any endpoint was named (i.e. discovery is not being relied on).
    pub fn is_explicit(&self) -> bool {
        !self.listen.is_empty() || !self.connect.is_empty()
    }

    /// Whether multicast scouting ends up enabled.
    pub fn multicast_enabled(&self) -> bool {
        self.multicast.unwrap_or(!self.is_explicit())
    }

    /// Apply to a Zenoh config. Endpoints are serialized as JSON, so an address
    /// containing a quote can't break out of the array.
    pub(crate) fn apply(&self, config: &mut zenoh::Config) -> Result<(), String> {
        for (field, eps) in [("listen", &self.listen), ("connect", &self.connect)] {
            if eps.is_empty() {
                continue;
            }
            let json = serde_json::to_string(eps)
                .map_err(|e| format!("zenoh {field} endpoints: {e}"))?;
            config
                .insert_json5(&format!("{field}/endpoints"), &json)
                .map_err(|e| format!("zenoh {field} endpoints {eps:?}: {e}"))?;
        }
        let _ = config.insert_json5(
            "scouting/multicast/enabled",
            if self.multicast_enabled() {
                "true"
            } else {
                "false"
            },
        );
        Ok(())
    }

    /// One-line description for a startup log.
    pub fn describe(&self) -> String {
        let mut parts = Vec::new();
        if !self.listen.is_empty() {
            parts.push(format!("listening on {}", self.listen.join(", ")));
        }
        if !self.connect.is_empty() {
            parts.push(format!("connecting to {}", self.connect.join(", ")));
        }
        if self.multicast_enabled() {
            parts.push("multicast discovery".to_string());
        }
        if parts.is_empty() {
            // Explicit endpoints with multicast forced off and none listed:
            // nothing would ever find this session.
            parts.push("no endpoints and no discovery".to_string());
        }
        parts.join(", ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn naming_an_endpoint_turns_discovery_off_but_not_irrevocably() {
        assert!(VizEndpoints::auto().multicast_enabled(), "LAN default");
        assert!(!VizEndpoints::connect(["tcp/127.0.0.1:7447"]).multicast_enabled());
        assert!(!VizEndpoints::listen(["tcp/0.0.0.0:7447"]).multicast_enabled());
        assert!(
            VizEndpoints::connect(["tcp/127.0.0.1:7447"])
                .with_multicast(true)
                .multicast_enabled(),
            "an explicit override wins over the rule"
        );
    }

    /// Listening and connecting are not exclusive: a node can accept viewers
    /// and dial a router at the same time.
    #[test]
    fn a_session_can_both_listen_and_connect() {
        let ep = VizEndpoints::listen(["tcp/0.0.0.0:7447"])
            .also_connect(["tcp/router:7447", "tcp/spare:7447"]);
        assert_eq!(ep.listen.len(), 1);
        assert_eq!(ep.connect.len(), 2);
        assert!(ep.describe().contains("listening on"));
        assert!(ep.describe().contains("connecting to"));
    }

    #[test]
    fn apply_sets_the_zenoh_config() {
        let mut config = zenoh::Config::default();
        VizEndpoints::connect(["tcp/127.0.0.1:7447"])
            .apply(&mut config)
            .unwrap();
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("tcp/127.0.0.1:7447"), "endpoint reached the config");
    }

    /// A configuration nothing could ever reach should say so rather than
    /// looking like a healthy setup in the log.
    #[test]
    fn an_unreachable_configuration_is_described_as_such() {
        let ep = VizEndpoints::auto().with_multicast(false);
        assert_eq!(ep.describe(), "no endpoints and no discovery");
    }
}
