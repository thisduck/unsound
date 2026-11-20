//! The Whisper `initial_prompt` that biases recognition toward the user's
//! vocabulary and, in meetings, the names of the people on the call. One
//! builder for dictation, meetings and the system-audio test.
//!
//! Kept short on purpose: long prompts cost accuracy on everything else, and
//! whisper.cpp keeps only the tail of an over-long prompt anyway.

use crate::settings::Settings;
use std::collections::HashSet;

/// Whole-prompt budget, roughly 150 tokens.
const MAX_PROMPT_CHARS: usize = 600;
/// People considered for a meeting prompt (each adds a full and a first name).
const MAX_PEOPLE: usize = 12;
/// Longer "names" are usually room or device labels, not people.
const MAX_NAME_CHARS: usize = 40;
const MAX_NAME_WORDS: usize = 4;

/// The user's corrected vocabulary, in dictionary order.
pub fn dictionary_terms(settings: &Settings) -> Vec<String> {
    settings
        .dictionary
        .iter()
        .map(|e| e.to.trim().to_string())
        .filter(|t| !t.is_empty())
        .collect()
}

/// Spellings to prime for a list of people: each full name, then its first
/// name. Emails, numbers and long labels are skipped.
pub fn name_terms<'a>(people: impl IntoIterator<Item = &'a str>) -> Vec<String> {
    let mut out = Vec::new();
    for name in people.into_iter().map(str::trim) {
        if !plausible_name(name) {
            continue;
        }
        if out.len() >= MAX_PEOPLE * 2 {
            break;
        }
        out.push(name.to_string());
        if let Some(first) = name.split_whitespace().next() {
            let first = first.trim_matches(|c: char| !c.is_alphanumeric());
            if first.chars().count() >= 2 && first != name {
                out.push(first.to_string());
            }
        }
    }
    out
}

fn plausible_name(name: &str) -> bool {
    !name.is_empty()
        && name.chars().count() <= MAX_NAME_CHARS
        && name.split_whitespace().count() <= MAX_NAME_WORDS
        && name.chars().any(char::is_alphabetic)
        && !name.chars().any(|c| c.is_ascii_digit() || c == '@' || c.is_control())
}

/// `"Glossary: a, b."`, deduplicated case-insensitively, names first, cut at
/// whole entries to fit the budget. `None` when there is nothing to prime.
pub fn prompt(names: &[String], terms: &[String]) -> Option<String> {
    let mut seen = HashSet::new();
    let mut entries: Vec<&str> = Vec::new();
    let mut len = "Glossary: .".len();
    for t in names.iter().chain(terms).map(|t| t.trim()) {
        if t.is_empty() || !seen.insert(t.to_lowercase()) {
            continue;
        }
        let add = t.len() + if entries.is_empty() { 0 } else { 2 };
        if len + add > MAX_PROMPT_CHARS {
            break;
        }
        len += add;
        entries.push(t);
    }
    (!entries.is_empty()).then(|| format!("Glossary: {}.", entries.join(", ")))
}

/// Dictation and file transcription: the dictionary only.
pub fn for_settings(settings: &Settings) -> Option<String> {
    prompt(&[], &dictionary_terms(settings))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn empty_inputs_send_no_prompt() {
        assert_eq!(prompt(&[], &[]), None);
        assert_eq!(prompt(&s(&["  "]), &s(&["", " "])), None);
    }

    #[test]
    fn dedupes_case_insensitively_and_keeps_first_spelling() {
        assert_eq!(
            prompt(&[], &s(&["Kubernetes", "kubernetes", " Priya ", "PRIYA"])).as_deref(),
            Some("Glossary: Kubernetes, Priya.")
        );
    }

    #[test]
    fn names_come_first_and_merge_with_the_dictionary() {
        let names = name_terms(["Priya Raman", "Marcus"]);
        assert_eq!(names, s(&["Priya Raman", "Priya", "Marcus"]));
        assert_eq!(
            prompt(&names, &s(&["priya", "Postgres"])).as_deref(),
            Some("Glossary: Priya Raman, Priya, Marcus, Postgres.")
        );
    }

    #[test]
    fn skips_emails_numbers_and_device_labels() {
        assert_eq!(
            name_terms([
                "priya@example.com",
                "+1 555 0100",
                "iPhone 15",
                "Conference Room Big Table North Wing",
                "",
                "Ana-Lucía Ortiz",
                "O'Neil",
            ]),
            s(&["Ana-Lucía Ortiz", "Ana-Lucía", "O'Neil"])
        );
    }

    #[test]
    fn caps_people_and_prompt_length() {
        let many: Vec<String> = (0..40).map(|i| format!("Person{} Surname", char::from(b'a' + (i % 26) as u8))).collect();
        let terms = name_terms(many.iter().map(String::as_str));
        assert!(terms.len() <= MAX_PEOPLE * 2);
        let long: Vec<String> = (0..200).map(|i| format!("term{i}")).collect();
        let p = prompt(&terms, &long).unwrap();
        assert!(p.len() <= MAX_PROMPT_CHARS, "{}", p.len());
        assert!(p.starts_with("Glossary: Persona Surname, Persona,"));
        assert!(p.ends_with('.'));
        // Cut at whole entries.
        assert!(!p.contains(", ."));
    }
}
