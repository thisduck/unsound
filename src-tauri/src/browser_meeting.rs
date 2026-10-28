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
    use core_foundation::array::CFArray;
    use core_foundation::base::{CFType, CFTypeID, CFTypeRef, TCFType};
    use core_foundation::string::{CFString, CFStringRef};
    use core_foundation::url::CFURL;
    use objc2_app_kit::NSWorkspace;
    use std::collections::VecDeque;
    use std::time::{Duration, Instant};

    #[link(name = "ApplicationServices", kind = "framework")]
    extern "C" {
        fn AXUIElementCreateApplication(pid: i32) -> CFTypeRef;
        fn AXUIElementGetTypeID() -> CFTypeID;
        fn AXUIElementCopyAttributeValue(
            element: CFTypeRef,
            name: CFStringRef,
            value: *mut CFTypeRef,
        ) -> i32;
        fn AXUIElementSetMessagingTimeout(element: CFTypeRef, timeout: f32) -> i32;
    }

    fn attribute(element: &CFType, name: &str) -> Option<CFType> {
        let name = CFString::new(name);
        let mut value = std::ptr::null();
        // SAFETY: type-check AX elements; a successful Copy returns a retained
        // CF object, released by CFType. Bound each cross-process request.
        unsafe {
            if element.type_of() != AXUIElementGetTypeID() {
                return None;
            }
            if AXUIElementSetMessagingTimeout(element.as_CFTypeRef(), 0.025) != 0 {
                return None;
            }
            if AXUIElementCopyAttributeValue(
                element.as_CFTypeRef(),
                name.as_concrete_TypeRef(),
                &mut value,
            ) != 0
                || value.is_null()
            {
                return None;
            }
            Some(CFType::wrap_under_create_rule(value))
        }
    }

    fn string(value: CFType) -> Option<String> {
        value
            .downcast::<CFString>()
            .map(|s| s.to_string())
            .or_else(|| {
                value
                    .downcast::<CFURL>()
                    .map(|url| url.get_string().to_string())
            })
    }

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
    // SAFETY: Create returns an owned AX element for this running process.
    let app = unsafe {
        let ptr = AXUIElementCreateApplication(front.processIdentifier());
        if ptr.is_null() {
            return None;
        }
        CFType::wrap_under_create_rule(ptr)
    };
    let window = attribute(&app, "AXFocusedWindow")?;
    if let Some(url) = attribute(&window, "AXDocument").and_then(string) {
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
        if attribute(&element, "AXRole").and_then(string).as_deref() == Some("AXWebArea") {
            return attribute(&element, "AXURL").and_then(string);
        }
        if depth >= 10 {
            continue;
        }
        if let Some(children) =
            attribute(&element, "AXChildren").and_then(|v| v.downcast::<CFArray>())
        {
            for child in children
                .iter()
                .take(64_usize.saturating_sub(visited + pending.len()))
            {
                if (*child).is_null() {
                    continue;
                }
                // SAFETY: AXChildren contains CF objects. Retain them before
                // the array is released; attribute() checks each object's type.
                let child = unsafe { CFType::wrap_under_get_rule(*child) };
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
