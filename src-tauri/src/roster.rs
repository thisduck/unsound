//! Who is in the call, read from the meeting app's own participant list
//! through Accessibility: Zoom's "Participants list", Teams' "Attendees", and
//! Google Meet's "Participants" panel in Chromium browsers.
//!
//! Read only. Nothing here opens a panel, clicks, focuses or sets anything; if
//! the panel is closed there is simply no roster. Missing permission, an
//! unknown app, a changed UI or a slow app all mean "no roster", never an
//! error. Reads run on their own thread, every few seconds, under a time and
//! node budget.
//!
//! The selectors and roster parsing follow Anarlog's MeetingAdapter.swift and
//! AXSnapshotReader.swift (https://github.com/fastrepl/anarlog), used under
//! the MIT License, Copyright (c) 2023-present Fastrepl, Inc.

use std::collections::HashMap;

/// One element of an Accessibility snapshot, flattened with its parent.
#[derive(Clone, Debug, Default)]
pub struct Node {
    pub parent: Option<usize>,
    pub role: String,
    /// AXValue for static text, else AXTitle.
    pub title: String,
    pub description: String,
    pub identifier: String,
    pub url: String,
}

impl Node {
    fn labels(&self) -> impl Iterator<Item = &str> {
        [self.title.as_str(), self.description.as_str()]
            .into_iter()
            .filter(|l| !l.is_empty())
    }
    fn has_label(&self, label: &str) -> bool {
        self.labels().any(|l| l == label)
    }
}

/// A bounded read of one app's call windows. `complete` is false when the
/// budget ran out or the app stopped answering: such a read decides nothing.
#[derive(Clone, Debug, Default)]
pub struct Snapshot {
    pub nodes: Vec<Node>,
    pub complete: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Platform {
    Zoom,
    Teams,
    Meet,
}

impl Platform {
    pub fn label(self) -> &'static str {
        match self {
            Platform::Zoom => "Zoom",
            Platform::Teams => "Teams",
            Platform::Meet => "Google Meet",
        }
    }
}

/// Chromium browsers whose Meet tab exposes Chrome's accessibility tree.
const CHROMIUM: &[&str] = &[
    "com.google.Chrome",
    "com.google.Chrome.beta",
    "com.google.Chrome.dev",
    "com.google.Chrome.canary",
    "org.chromium.Chromium",
    "company.thebrowser.Browser",
    "com.microsoft.edgemac",
    "com.brave.Browser",
];

pub fn platform(bundle_id: &str) -> Option<Platform> {
    match bundle_id {
        "us.zoom.xos" => Some(Platform::Zoom),
        "com.microsoft.teams2" => Some(Platform::Teams),
        id if CHROMIUM.contains(&id) => Some(Platform::Meet),
        _ => None,
    }
}

/// One roster row. `me` is `None` when the app doesn't mark the local user.
#[derive(Clone, Debug, PartialEq)]
pub struct Entry {
    pub name: String,
    pub me: Option<bool>,
}

/// What one app's windows show.
#[derive(Clone, Debug, PartialEq)]
pub enum Reading {
    /// No call confirmed (or the read was incomplete).
    NoCall,
    /// A call; `people` is empty when its participant panel is closed.
    Call(Vec<Entry>),
}

/// Pages the Teams app shows calls in.
const TEAMS_HOSTS: &[&str] = &["teams.live.com", "teams.microsoft.com", "teams.cloud.microsoft"];

fn is_teams_page(url: &str) -> bool {
    reqwest::Url::parse(url).is_ok_and(|u| {
        u.scheme() == "https" && u.host_str().is_some_and(|h| TEAMS_HOSTS.contains(&h))
    })
}

fn is_meet_call(url: &str) -> bool {
    crate::browser_meeting::provider(url) == Some("Google Meet")
}

/// Whether a web area may be read at all: only call pages, never other tabs.
fn web_area_allowed(platform: Platform, url: &str) -> bool {
    match platform {
        Platform::Teams => is_teams_page(url),
        Platform::Meet => is_meet_call(url),
        Platform::Zoom => false,
    }
}

/// Role markers that may trail a display name. `me`/`you` mark the local user.
const MARKERS: &[&str] = &[
    "me", "you", "host", "co-host", "cohost", "guest", "organizer", "organiser", "presenter",
    "external", "unverified", "moderator",
];

/// Strip trailing role markers like " (Host, me)" or " (You)" from a roster
/// label. Returns the name and whether a marker said it is the local user;
/// `None` for labels that aren't plausibly a person's name.
pub fn clean_name(label: &str) -> Option<(String, bool)> {
    let mut name = label.trim();
    let mut me = false;
    while name.ends_with(')') {
        let Some(open) = name.rfind('(') else { break };
        let inner = &name[open + 1..name.len() - 1];
        let tokens: Vec<String> = inner.split(',').map(|t| t.trim().to_lowercase()).collect();
        if tokens.is_empty() || !tokens.iter().all(|t| MARKERS.contains(&t.as_str())) {
            break;
        }
        me |= tokens.iter().any(|t| t == "me" || t == "you");
        name = name[..open].trim_end();
    }
    let ok = !name.is_empty()
        && name.chars().count() <= 100
        && name.chars().any(char::is_alphanumeric)
        && !name.chars().any(char::is_control);
    ok.then(|| (name.to_string(), me))
}

struct Tree<'a> {
    nodes: &'a [Node],
    children: Vec<Vec<usize>>,
}

impl<'a> Tree<'a> {
    fn new(nodes: &'a [Node]) -> Self {
        let mut children = vec![Vec::new(); nodes.len()];
        for (i, n) in nodes.iter().enumerate() {
            if let Some(p) = n.parent.filter(|&p| p < nodes.len() && p != i) {
                children[p].push(i);
            }
        }
        Self { nodes, children }
    }

    /// Every node below `root` (not `root` itself), depth first.
    fn descendants(&self, root: usize) -> Vec<usize> {
        let mut out = Vec::new();
        let mut stack: Vec<usize> = self.children[root].iter().rev().copied().collect();
        while let Some(i) = stack.pop() {
            if out.len() > self.nodes.len() {
                break; // malformed parent links
            }
            out.push(i);
            stack.extend(self.children[i].iter().rev());
        }
        out
    }

    fn node(&self, i: usize) -> &Node {
        &self.nodes[i]
    }

    fn find(&self, within: &[usize], pred: impl Fn(&Node) -> bool) -> Vec<usize> {
        within.iter().copied().filter(|&i| pred(self.node(i))).collect()
    }
}

/// The call and roster one app's snapshot shows. Pure, so the selectors are
/// testable against synthetic trees.
pub fn parse(platform: Platform, snapshot: &Snapshot) -> Reading {
    if !snapshot.complete {
        return Reading::NoCall;
    }
    let tree = Tree::new(&snapshot.nodes);
    let all: Vec<usize> = (0..snapshot.nodes.len()).collect();
    let people = match platform {
        Platform::Zoom => zoom(&tree, &all),
        Platform::Meet => meet(&tree, &all),
        Platform::Teams => teams(&tree, &all),
    };
    match people {
        Some(people) => Reading::Call(dedupe(people)),
        None => Reading::NoCall,
    }
}

fn dedupe(people: Vec<Entry>) -> Vec<Entry> {
    let mut out: Vec<Entry> = Vec::new();
    for e in people {
        match out.iter_mut().find(|o| o.name.to_lowercase() == e.name.to_lowercase()) {
            // Two rows with one name: keep one; either can carry the self mark.
            Some(o) => o.me = match (o.me, e.me) {
                (Some(a), Some(b)) => Some(a || b),
                (a, b) => a.or(b),
            },
            None => out.push(e),
        }
    }
    out
}

/// `Some(roster)` for exactly one confirmed call, else `None`.
fn zoom(tree: &Tree, all: &[usize]) -> Option<Vec<Entry>> {
    let windows = tree.find(all, |n| n.role == "AXWindow" && n.identifier == "zm.meeting.window.main");
    let calls: Vec<usize> = windows
        .into_iter()
        .filter(|&w| {
            let buttons = tree.find(&tree.descendants(w), |n| n.role == "AXButton");
            ["leave", "audio", "participant"]
                .iter()
                .all(|id| buttons.iter().any(|&b| tree.node(b).identifier == *id))
        })
        .collect();
    let [window] = calls[..] else { return None };
    let content = tree.descendants(window);
    let lists = tree.find(&content, |n| n.role == "AXOutline" && n.has_label("Participants list"));
    let mut roster = Vec::new();
    if let [list] = lists[..] {
        for row in tree.find(&tree.descendants(list), |n| n.role == "AXRow") {
            let texts = tree.find(&tree.descendants(row), |n| n.role == "AXStaticText");
            let [text] = texts[..] else { continue };
            let mut labels: Vec<&str> = tree.node(text).labels().collect();
            labels.dedup();
            let [label] = labels[..] else { continue };
            if label.len() > 256 {
                continue;
            }
            if let Some((name, me)) = clean_name(label) {
                roster.push(Entry { name, me: Some(me) });
            }
        }
    }
    Some(roster)
}

fn meet(tree: &Tree, all: &[usize]) -> Option<Vec<Entry>> {
    let calls: Vec<usize> = tree
        .find(all, |n| n.role == "AXWebArea" && is_meet_call(&n.url))
        .into_iter()
        .filter(|&area| {
            let content: Vec<&Node> = tree.descendants(area).into_iter().map(|i| tree.node(i)).collect();
            content.iter().any(|n| n.role == "AXButton" && n.has_label("Leave call"))
                && content.iter().any(|n| n.has_label("Call controls"))
                && content.iter().any(|n| {
                    n.role == "AXButton"
                        && (n.has_label("Turn on microphone") || n.has_label("Turn off microphone"))
                })
        })
        .collect();
    let [area] = calls[..] else { return None };
    let content = tree.descendants(area);
    let lists = tree.find(&content, |n| n.role == "AXList" && n.has_label("Participants"));
    let mut roster = Vec::new();
    if let [list] = lists[..] {
        for &row in &tree.children[list] {
            let r = tree.node(row);
            if r.role != "AXGroup" {
                continue;
            }
            let mut labels: Vec<&str> = r.labels().collect();
            labels.dedup();
            let [label] = labels[..] else { continue };
            if label.len() > 256 {
                continue;
            }
            let inside = tree.descendants(row);
            let you = format!("{label} (You)");
            let names = tree.find(&inside, |n| {
                n.role == "AXStaticText" && (n.has_label(label) || n.has_label(&you))
            });
            let [name_node] = names[..] else { continue };
            let me = tree.node(name_node).has_label(&you)
                || inside
                    .iter()
                    .any(|&i| tree.node(i).role == "AXStaticText" && tree.node(i).has_label("(You)"));
            if let Some((name, marked)) = clean_name(label) {
                roster.push(Entry { name, me: Some(me || marked) });
            }
        }
    }
    Some(roster)
}

fn teams(tree: &Tree, all: &[usize]) -> Option<Vec<Entry>> {
    let calls: Vec<usize> = tree
        .find(all, |n| n.role == "AXWebArea" && is_teams_page(&n.url))
        .into_iter()
        .filter(|&area| {
            let content = tree.descendants(area);
            let toolbars = tree.find(&content, |n| n.role == "AXToolbar" && n.has_label("Meeting controls"));
            let [toolbar] = toolbars[..] else { return false };
            let controls: Vec<&Node> = tree.descendants(toolbar).into_iter().map(|i| tree.node(i)).collect();
            let content: Vec<&Node> = content.into_iter().map(|i| tree.node(i)).collect();
            controls.iter().any(|n| n.role == "AXButton" && n.has_label("People"))
                && controls
                    .iter()
                    .any(|n| n.role == "AXButton" && (n.has_label("Unmute mic") || n.has_label("Mute mic")))
                && content.iter().any(|n| n.role == "AXButton" && n.has_label("Leave"))
                && content.iter().any(|n| n.has_label("Calling controls"))
        })
        .collect();
    let [area] = calls[..] else { return None };
    let content = tree.descendants(area);
    let lists = tree.find(&content, |n| n.role == "AXOutline" && n.has_label("Attendees"));
    let mut roster = Vec::new();
    if let [list] = lists[..] {
        for row in tree.find(&tree.descendants(list), |n| n.role == "AXRow") {
            let r = tree.node(row);
            // A row reads "<name>, <status>…"; its name is the text it starts with.
            let starts_row = |l: &str| {
                !l.trim().is_empty() && l.len() <= 256 && r.labels().any(|rl| rl.starts_with(&format!("{l}, ")))
            };
            let texts = tree.find(&tree.descendants(row), |n| {
                n.role == "AXStaticText" && n.labels().any(starts_row)
            });
            let [text] = texts[..] else { continue };
            let mut labels: Vec<&str> = tree.node(text).labels().filter(|l| starts_row(l)).collect();
            labels.dedup();
            let [label] = labels[..] else { continue };
            if let Some((name, marked)) = clean_name(label) {
                // Teams doesn't mark the local user; a marker is a bonus.
                roster.push(Entry { name, me: marked.then_some(true) });
            }
        }
    }
    Some(roster)
}

/// The roster of the one app showing a call: `(app label, people)`. Several
/// apps in a call at once is ambiguous and yields nothing.
pub fn sole_call(readings: Vec<(Platform, Reading)>) -> Option<(Platform, Vec<Entry>)> {
    let mut calls = readings.into_iter().filter_map(|(p, r)| match r {
        Reading::Call(people) => Some((p, people)),
        Reading::NoCall => None,
    });
    let first = calls.next()?;
    if calls.next().is_some() {
        return None;
    }
    (!first.1.is_empty()).then_some(first)
}

// ---- reading the live apps ---------------------------------------------------

#[cfg(target_os = "macos")]
mod reader {
    use super::*;
    use crate::ax;
    use core_foundation::base::CFType;
    use std::time::{Duration, Instant};

    const TIMEOUT: f32 = 0.04;
    const BUDGET: Duration = Duration::from_millis(750);
    const MAX_NODES: usize = 2_000;
    const MAX_DEPTH: usize = 40;
    const MAX_TEXT: usize = 512;

    struct Walk {
        platform: Platform,
        deadline: Instant,
        nodes: Vec<Node>,
        complete: bool,
    }

    impl Walk {
        fn unreliable(&mut self, err: i32) {
            if matches!(err, ax::ERR_CANNOT_COMPLETE | ax::ERR_INVALID_ELEMENT | ax::ERR_API_DISABLED) {
                self.complete = false;
            }
        }

        fn text(value: Option<CFType>) -> String {
            value
                .and_then(ax::string)
                .filter(|s| s.len() <= MAX_TEXT)
                .unwrap_or_default()
        }

        fn walk(&mut self, element: &CFType, parent: Option<usize>, depth: usize) {
            if !self.complete {
                return;
            }
            if self.nodes.len() >= MAX_NODES || depth >= MAX_DEPTH || Instant::now() >= self.deadline {
                self.complete = false;
                return;
            }
            let names = ["AXRole", "AXTitle", "AXDescription", "AXIdentifier", "AXChildren"];
            let mut values = match ax::attributes(element, &names, TIMEOUT) {
                Ok(v) => v.into_iter(),
                Err(e) => {
                    self.unreliable(e);
                    return;
                }
            };
            let mut next = || values.next().flatten();
            let role = Self::text(next());
            let mut title = Self::text(next());
            let description = Self::text(next());
            let identifier = Self::text(next());
            let children = next();
            let mut url = String::new();
            if role == "AXWebArea" {
                url = Self::text(ax::attribute(element, "AXURL", TIMEOUT));
                // Other pages are skipped unread.
                if !web_area_allowed(self.platform, &url) {
                    return;
                }
            }
            // Chat history is never read.
            if title == "Message List" || description == "Message List" {
                return;
            }
            // Only static labels; never editable values, passwords or chat input.
            if role == "AXStaticText" {
                let value = Self::text(ax::attribute(element, "AXValue", TIMEOUT));
                if !value.is_empty() {
                    title = value;
                }
            }
            let index = self.nodes.len();
            self.nodes.push(Node { parent, role: role.clone(), title, description, identifier, url });
            if matches!(role.as_str(), "AXTextField" | "AXTextArea" | "AXSecureTextField") {
                return;
            }
            for child in children.map(ax::elements).unwrap_or_default() {
                self.walk(&child, Some(index), depth + 1);
                if !self.complete {
                    return;
                }
            }
        }
    }

    pub fn snapshot(pid: i32, platform: Platform) -> Snapshot {
        let Some(app) = ax::application(pid) else {
            return Snapshot::default();
        };
        let mut walk = Walk {
            platform,
            deadline: Instant::now() + BUDGET,
            nodes: Vec::new(),
            complete: true,
        };
        let mut windows = match ax::copy(&app, "AXWindows", TIMEOUT) {
            Ok(v) => ax::elements(v),
            Err(_) => return Snapshot::default(),
        };
        // An app on another Space can list no windows while its main one answers.
        if windows.is_empty() {
            for name in ["AXMainWindow", "AXFocusedWindow"] {
                if let Some(w) = ax::attribute(&app, name, TIMEOUT).filter(ax::is_element) {
                    windows.push(w);
                }
            }
        }
        for window in windows {
            // Zoom: only the meeting window, never Home or chat windows.
            if platform == Platform::Zoom
                && ax::attribute(&window, "AXIdentifier", TIMEOUT)
                    .and_then(ax::string)
                    .as_deref()
                    != Some("zm.meeting.window.main")
            {
                continue;
            }
            walk.walk(&window, None, 0);
            if !walk.complete {
                break;
            }
        }
        Snapshot { complete: walk.complete, nodes: walk.nodes }
    }

    /// Running supported apps as `(pid, bundle id, platform)`, preferring the
    /// ones holding the microphone when CoreAudio can say.
    fn candidates() -> Vec<(i32, String, Platform)> {
        use objc2_app_kit::NSWorkspace;
        let running: Vec<(i32, String, Platform)> = NSWorkspace::sharedWorkspace()
            .runningApplications()
            .iter()
            .filter_map(|app| {
                let id = app.bundleIdentifier()?.to_string();
                Some((app.processIdentifier(), id.clone(), platform(&id)?))
            })
            .collect();
        let Some(holders) = crate::detect::mic_users() else {
            return running;
        };
        let on_mic: Vec<_> = running
            .iter()
            .filter(|(_, id, _)| {
                holders.iter().any(|h| {
                    h.bundle_id == *id || h.bundle_id.strip_prefix(id.as_str()).is_some_and(|r| r.starts_with('.'))
                })
            })
            .cloned()
            .collect();
        if on_mic.is_empty() { running } else { on_mic }
    }

    pub fn read() -> Option<(Platform, Vec<Entry>)> {
        if !crate::permissions::status().accessibility {
            return None;
        }
        let readings = candidates()
            .into_iter()
            .map(|(pid, _, p)| (p, parse(p, &snapshot(pid, p))))
            .collect();
        sole_call(readings)
    }
}

/// The current call's roster, or `None` whenever it can't be read.
#[cfg(target_os = "macos")]
pub fn read() -> Option<(Platform, Vec<Entry>)> {
    reader::read()
}

#[cfg(not(target_os = "macos"))]
pub fn read() -> Option<(Platform, Vec<Entry>)> {
    None
}

// ---- polling during a meeting ------------------------------------------------

/// Gentle: a roster changes on joins and leaves, not by the second.
const POLL: std::time::Duration = std::time::Duration::from_secs(5);

/// Poll the roster for one meeting until `stop`, keeping the union of who was
/// seen on the meeting (see `participants`). Runs on its own thread.
pub fn spawn(app: tauri::AppHandle, meeting_id: String, stop: std::sync::Arc<std::sync::atomic::AtomicBool>) {
    use std::sync::atomic::Ordering;
    use tauri::{Emitter, Manager};
    let spawned = std::thread::Builder::new()
        .name("meeting-roster".into())
        .spawn(move || {
            let db = app.state::<crate::store::Db>();
            let self_names = crate::store::known_self_names(&db).unwrap_or_default();
            let mut observer = crate::participants::Observer::new(
                crate::store::meeting_people(&db, &meeting_id)
                    .map(|(_, p)| p)
                    .unwrap_or_default(),
                self_names,
            );
            let mut last_names: HashMap<String, bool> = HashMap::new();
            let mut next = std::time::Instant::now() + POLL;
            while !stop.load(Ordering::Relaxed) {
                std::thread::sleep(std::time::Duration::from_millis(250));
                if std::time::Instant::now() < next {
                    continue;
                }
                next = std::time::Instant::now() + POLL;
                let Some(at_ms) = crate::meeting::position_ms(&app) else { continue };
                let roster = read();
                if !observer.observe(roster.as_ref().map(|(p, e)| (p.label(), e.as_slice())), at_ms) {
                    continue;
                }
                if crate::store::set_participants(&db, &meeting_id, &observer.record).is_err() {
                    continue;
                }
                let names: HashMap<String, bool> =
                    observer.record.people.iter().map(|p| (p.name.clone(), p.me)).collect();
                if names != last_names {
                    last_names = names;
                    let _ = app.emit(
                        "meeting-participants",
                        serde_json::json!({ "meetingId": meeting_id, "participants": observer.record.people }),
                    );
                }
            }
        });
    if let Err(e) = spawned {
        eprintln!("[roster] not started: {e}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a snapshot from (depth, role, title, description, identifier, url).
    fn tree(rows: &[(usize, &str, &str, &str, &str, &str)]) -> Snapshot {
        let mut nodes: Vec<Node> = Vec::new();
        let mut stack: Vec<usize> = Vec::new();
        for &(depth, role, title, description, identifier, url) in rows {
            stack.truncate(depth);
            nodes.push(Node {
                parent: stack.last().copied(),
                role: role.into(),
                title: title.into(),
                description: description.into(),
                identifier: identifier.into(),
                url: url.into(),
            });
            stack.push(nodes.len() - 1);
        }
        Snapshot { nodes, complete: true }
    }

    fn entry(name: &str, me: Option<bool>) -> Entry {
        Entry { name: name.into(), me }
    }

    fn zoom_tree(rows: &[&str], with_panel: bool) -> Snapshot {
        let mut t: Vec<(usize, &str, &str, &str, &str, &str)> = vec![
            (0, "AXWindow", "Zoom Meeting", "", "zm.meeting.window.main", ""),
            (1, "AXGroup", "", "", "", ""),
            (2, "AXButton", "Leave", "", "leave", ""),
            (2, "AXButton", "Mute", "", "audio", ""),
            (2, "AXButton", "Participants", "", "participant", ""),
        ];
        if with_panel {
            t.push((1, "AXOutline", "", "Participants list", "", ""));
            for r in rows {
                t.push((2, "AXRow", "", "", "", ""));
                t.push((3, "AXCell", "", "", "", ""));
                t.push((4, "AXStaticText", r, "", "", ""));
                t.push((4, "AXImage", "", "", "", ""));
            }
        }
        tree(&t)
    }

    #[test]
    fn cleans_role_markers_and_detects_self() {
        assert_eq!(clean_name("Adnan Ali (Host, me)"), Some(("Adnan Ali".into(), true)));
        assert_eq!(clean_name("Adnan (me)"), Some(("Adnan".into(), true)));
        assert_eq!(clean_name("Priya Raman (Host)"), Some(("Priya Raman".into(), false)));
        assert_eq!(clean_name("Guest User (Guest)"), Some(("Guest User".into(), false)));
        assert_eq!(clean_name("Marcus (You)"), Some(("Marcus".into(), true)));
        assert_eq!(clean_name("Lee (Co-host) (Guest)"), Some(("Lee".into(), false)));
        assert_eq!(clean_name("  Dana  (Organizer)  "), Some(("Dana".into(), false)));
        // Parentheses that are part of the name stay.
        assert_eq!(clean_name("Sam (Design)"), Some(("Sam (Design)".into(), false)));
        assert_eq!(clean_name("(me)"), None);
        assert_eq!(clean_name("   "), None);
        assert_eq!(clean_name("--"), None);
    }

    #[test]
    fn zoom_roster_with_self_and_host_markers() {
        let s = zoom_tree(&["Adnan Ali (Host, me)", "Priya Raman", "Marcus (Guest)"], true);
        assert_eq!(
            parse(Platform::Zoom, &s),
            Reading::Call(vec![
                entry("Adnan Ali", Some(true)),
                entry("Priya Raman", Some(false)),
                entry("Marcus", Some(false)),
            ])
        );
    }

    #[test]
    fn zoom_closed_panel_is_a_call_without_roster() {
        let s = zoom_tree(&[], false);
        assert_eq!(parse(Platform::Zoom, &s), Reading::Call(vec![]));
        assert_eq!(sole_call(vec![(Platform::Zoom, parse(Platform::Zoom, &s))]), None);
    }

    #[test]
    fn zoom_needs_call_controls_and_a_complete_read() {
        // Home window: no meeting controls.
        let home = tree(&[
            (0, "AXWindow", "Zoom Workplace", "", "zm.home.window", ""),
            (1, "AXOutline", "", "Participants list", "", ""),
            (2, "AXRow", "", "", "", ""),
            (3, "AXStaticText", "Priya", "", "", ""),
        ]);
        assert_eq!(parse(Platform::Zoom, &home), Reading::NoCall);
        let mut partial = zoom_tree(&["Priya"], true);
        partial.complete = false;
        assert_eq!(parse(Platform::Zoom, &partial), Reading::NoCall);
        // Leave and audio without the participants button is not a call.
        let no_participant = tree(&[
            (0, "AXWindow", "", "", "zm.meeting.window.main", ""),
            (1, "AXButton", "", "", "leave", ""),
            (1, "AXButton", "", "", "audio", ""),
        ]);
        assert_eq!(parse(Platform::Zoom, &no_participant), Reading::NoCall);
    }

    #[test]
    fn zoom_skips_ambiguous_rows() {
        let s = tree(&[
            (0, "AXWindow", "", "", "zm.meeting.window.main", ""),
            (1, "AXButton", "", "", "leave", ""),
            (1, "AXButton", "", "", "audio", ""),
            (1, "AXButton", "", "", "participant", ""),
            (1, "AXOutline", "", "Participants list", "", ""),
            // Two texts in one row: unclear which is the name.
            (2, "AXRow", "", "", "", ""),
            (3, "AXStaticText", "Priya", "", "", ""),
            (3, "AXStaticText", "Muted", "", "", ""),
            // Title and description disagree.
            (2, "AXRow", "", "", "", ""),
            (3, "AXStaticText", "Marcus", "Someone else", "", ""),
            (2, "AXRow", "", "", "", ""),
            (3, "AXStaticText", "Lee", "Lee", "", ""),
        ]);
        assert_eq!(parse(Platform::Zoom, &s), Reading::Call(vec![entry("Lee", Some(false))]));
    }

    fn meet_tree(url: &str, rows: &[(&str, &str)]) -> Snapshot {
        let mut t: Vec<(usize, &str, &str, &str, &str, &str)> = vec![
            (0, "AXWindow", "Meet", "", "", ""),
            (1, "AXWebArea", "Meet", "", "", url),
            (2, "AXGroup", "", "Call controls", "", ""),
            (3, "AXButton", "", "Leave call", "", ""),
            (3, "AXButton", "", "Turn off microphone", "", ""),
            (2, "AXList", "", "Participants", "", ""),
        ];
        for (label, text) in rows {
            t.push((3, "AXGroup", label, "", "", ""));
            t.push((4, "AXStaticText", text, "", "", ""));
        }
        tree(&t)
    }

    #[test]
    fn meet_roster_marks_you() {
        let s = meet_tree(
            "https://meet.google.com/abc-defg-hij",
            &[("Adnan Ali", "Adnan Ali (You)"), ("Priya Raman", "Priya Raman")],
        );
        assert_eq!(
            parse(Platform::Meet, &s),
            Reading::Call(vec![entry("Adnan Ali", Some(true)), entry("Priya Raman", Some(false))])
        );
    }

    #[test]
    fn meet_requires_a_call_url_and_controls() {
        let s = meet_tree("https://meet.google.com/landing", &[("Priya", "Priya")]);
        assert_eq!(parse(Platform::Meet, &s), Reading::NoCall);
        let no_controls = tree(&[
            (0, "AXWindow", "", "", "", ""),
            (1, "AXWebArea", "", "", "", "https://meet.google.com/abc-defg-hij"),
            (2, "AXList", "", "Participants", "", ""),
            (3, "AXGroup", "Priya", "", "", ""),
            (4, "AXStaticText", "Priya", "", "", ""),
        ]);
        assert_eq!(parse(Platform::Meet, &no_controls), Reading::NoCall);
    }

    #[test]
    fn teams_roster_reads_names_from_row_labels() {
        let s = tree(&[
            (0, "AXWindow", "", "", "", ""),
            (1, "AXWebArea", "", "", "", "https://teams.microsoft.com/v2/"),
            (2, "AXGroup", "", "Calling controls", "", ""),
            (3, "AXButton", "", "Leave", "", ""),
            (2, "AXToolbar", "", "Meeting controls", "", ""),
            (3, "AXButton", "", "People", "", ""),
            (3, "AXButton", "", "Mute mic", "", ""),
            (2, "AXOutline", "", "Attendees", "", ""),
            (3, "AXRow", "", "Priya Raman, Organizer, Muted", "", ""),
            (4, "AXStaticText", "Priya Raman", "", "", ""),
            (4, "AXStaticText", "Organizer", "", "", ""),
            (3, "AXRow", "", "Marcus Lee (Guest), External", "", ""),
            (4, "AXStaticText", "Marcus Lee (Guest)", "", "", ""),
            (3, "AXRow", "", "Unlabelled row", "", ""),
            (4, "AXStaticText", "Unlabelled row", "", "", ""),
        ]);
        assert_eq!(
            parse(Platform::Teams, &s),
            Reading::Call(vec![entry("Priya Raman", None), entry("Marcus Lee", None)])
        );
        // A non-Teams page is never a Teams call.
        let mut other = s.clone();
        other.nodes[1].url = "https://example.com/".into();
        assert_eq!(parse(Platform::Teams, &other), Reading::NoCall);
    }

    #[test]
    fn several_calls_at_once_are_ambiguous() {
        let z = parse(Platform::Zoom, &zoom_tree(&["Priya"], true));
        let m = parse(
            Platform::Meet,
            &meet_tree("https://meet.google.com/abc-defg-hij", &[("Marcus", "Marcus")]),
        );
        assert_eq!(sole_call(vec![(Platform::Zoom, z.clone()), (Platform::Meet, m)]), None);
        assert_eq!(
            sole_call(vec![(Platform::Zoom, z), (Platform::Meet, Reading::NoCall)]),
            Some((Platform::Zoom, vec![entry("Priya", Some(false))]))
        );
    }

    #[test]
    fn duplicate_rows_collapse_and_malformed_parents_do_not_loop() {
        let s = zoom_tree(&["Priya", "priya", "Adnan (me)"], true);
        assert_eq!(
            parse(Platform::Zoom, &s),
            Reading::Call(vec![entry("Priya", Some(false)), entry("Adnan", Some(true))])
        );
        let mut looped = zoom_tree(&["Priya"], true);
        looped.nodes[0].parent = Some(1);
        let _ = parse(Platform::Zoom, &looped);
    }

    #[test]
    fn platforms_by_bundle_id() {
        assert_eq!(platform("us.zoom.xos"), Some(Platform::Zoom));
        assert_eq!(platform("com.microsoft.teams2"), Some(Platform::Teams));
        assert_eq!(platform("company.thebrowser.Browser"), Some(Platform::Meet));
        assert_eq!(platform("com.apple.Safari"), None);
        assert_eq!(platform("com.microsoft.teams"), None);
    }
}
