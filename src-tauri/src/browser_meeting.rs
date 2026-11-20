//! Browser microphone activity alone is not evidence of a meeting. Only a
//! recognized meeting URL in the foreground browser can qualify for a prompt.
//! Unknown pages and unavailable Accessibility data deliberately stay silent.

pub fn provider(url: &str) -> Option<&'static str> {
    let url = reqwest::Url::parse(url).ok()?;
    if url.scheme() != "https" || !url.username().is_empty() || url.password().is_some() {
        return None;
    }
    let host = url.host_str()?;
    let path: Vec<_> = url.path().trim_matches('/').split('/').collect();
    let numeric_id = |id: &str| !id.is_empty() && id.bytes().all(|c| c.is_ascii_digit());
    match host {
        "meet.google.com" => {
            let [code] = path.as_slice() else { return None };
            let parts: Vec<_> = code.split('-').collect();
            (parts.len() == 3
                && parts.iter().map(|p| p.len()).eq([3, 4, 3])
                && parts
                    .iter()
                    .all(|p| p.bytes().all(|c| c.is_ascii_lowercase())))
            .then_some("Google Meet")
        }
        "teams.microsoft.com" | "teams.live.com" | "teams.cloud.microsoft" => match path.as_slice()
        {
            ["l", "meetup-join", meeting, ..] if !meeting.is_empty() => Some("Teams"),
            ["meet", id] if numeric_id(id) => Some("Teams"),
            _ => None,
        },
        host if host == "zoom.us" || host.ends_with(".zoom.us") => match path.as_slice() {
            ["wc", id, "join" | "start"] | ["wc", "join", id] if numeric_id(id) => Some("Zoom"),
            _ => None,
        },
        _ => None,
    }
}

/// Read only the foreground browser's current document, never background tabs
/// or browsing history. Permission is checked without opening an OS prompt.
#[cfg(target_os = "macos")]
pub fn foreground_url(browser_id: &str) -> Option<String> {
    use crate::ax;
    use objc2_app_kit::NSWorkspace;
    use std::collections::VecDeque;
    use std::time::{Duration, Instant};

    const TIMEOUT: f32 = 0.025;
    let text = |element: &core_foundation::base::CFType, name: &str| ax::attribute(element, name, TIMEOUT).and_then(ax::string);

    if !crate::permissions::status().accessibility {
        return None;
    }
    let front = NSWorkspace::sharedWorkspace().frontmostApplication()?;
    let bundle = front.bundleIdentifier()?.to_string();
    if bundle != browser_id
        && !bundle
            .strip_prefix(browser_id)
            .is_some_and(|s| s.starts_with('.'))
    {
        return None;
    }
    let app = ax::application(front.processIdentifier())?;
    let window = ax::attribute(&app, "AXFocusedWindow", TIMEOUT)?;
    if let Some(url) = text(&window, "AXDocument") {
        return Some(url);
    }
    // Some browsers expose the selected page URL on AXWebArea instead of the
    // window. Stop at the first web area; do not scan iframe content or links.
    let deadline = Instant::now() + Duration::from_millis(250);
    let mut pending = VecDeque::from([(window, 0)]);
    let mut visited = 0;
    while let Some((element, depth)) = pending.pop_front() {
        if visited >= 64 || Instant::now() >= deadline {
            break;
        }
        visited += 1;
        if text(&element, "AXRole").as_deref() == Some("AXWebArea") {
            return text(&element, "AXURL");
        }
        if depth >= 10 {
            continue;
        }
        if let Some(children) = ax::attribute(&element, "AXChildren", TIMEOUT) {
            let room = 64_usize.saturating_sub(visited + pending.len());
            for child in ax::elements(children).into_iter().take(room) {
                pending.push_back((child, depth + 1));
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognizes_only_meeting_routes_on_known_services() {
        for (url, expected) in [
            (
                "https://meet.google.com/abc-defg-hij?authuser=1",
                "Google Meet",
            ),
            ("https://us02web.zoom.us/wc/123456789/join", "Zoom"),
            ("https://app.zoom.us/wc/join/123456789", "Zoom"),
            (
                "https://teams.microsoft.com/l/meetup-join/19%3ameeting_123/0?context=x",
                "Teams",
            ),
            ("https://teams.live.com/meet/123456789", "Teams"),
        ] {
            assert_eq!(provider(url), Some(expected), "{url}");
        }
    }

    #[test]
    fn transcription_sites_homepages_and_lookalike_urls_never_qualify() {
        for url in [
            "https://otter.ai/record",
            "https://app.granola.ai/",
            "https://chatgpt.com/",
            "https://dictation.io/speech",
            "https://docs.google.com/document/d/123/edit",
            "https://meet.google.com/",
            "https://meet.google.com/landing",
            "https://meet.google.com/lookup/room",
            "https://zoom.us/test",
            "https://zoom.us/profile",
            "https://teams.microsoft.com/v2/",
            "https://teams.microsoft.com/l/chat/123",
            "https://example.com/?url=https://meet.google.com/abc-defg-hij",
            "https://meet.google.com.evil.test/abc-defg-hij",
            "https://fakezoom.us/wc/123/join",
            "https://meet.google.com@evil.test/abc-defg-hij",
            "https://user@meet.google.com/abc-defg-hij",
            "http://meet.google.com/abc-defg-hij",
            "Google Meet",
            "",
        ] {
            assert_eq!(provider(url), None, "{url}");
        }
    }
}
