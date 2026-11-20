//! Cleanup of speech-model output: non-speech tags, stock phrases Whisper emits
//! for silence, and decoder loops. Words are Unicode runs between punctuation
//! and whitespace, so this works the same for non-Latin scripts.
use std::collections::HashMap;

const NO_SPEECH_PROBABILITY_THRESHOLD: f32 = 0.6;

/// Whole-segment phrases Whisper produces for silence or noise, in the
/// normalized form `words` yields. Dropped only when Whisper itself rates the
/// segment as probably not speech, so the same words spoken aloud survive.
const SILENCE_PHRASES: &[&str] = &[
    "you",
    "thank you",
    "thanks",
    "thank you very much",
    "thank you so much",
    "thank you for watching",
    "thanks for watching",
    "thank you for listening",
    "thanks for listening",
    "please subscribe",
    "like and subscribe",
    "bye",
    "bye bye",
    "ご視聴ありがとうございました",
    "시청해주셔서 감사합니다",
    "продолжение следует",
    "vielen dank fürs zuschauen",
    "gracias por ver el video",
    "merci d avoir regardé",
    "obrigado por assistir",
];

/// Subtitle credits from Whisper's training data, matched as segment prefixes
/// under the same no-speech gate.
const CREDIT_PREFIXES: &[&str] = &[
    "subtitles by",
    "subtitled by",
    "captions by",
    "captioned by",
    "transcription by",
    "transcribed by",
    "translated by",
    "untertitel im auftrag",
    "sous titres réalisés",
];

/// Parenthesized annotations removed wherever they appear. Square-bracket
/// annotations are always removed, as Whisper never puts speech in them.
const NON_SPEECH_TAGS: &[&str] = &[
    "applause",
    "background noise",
    "beep",
    "blank audio",
    "coughing",
    "coughs",
    "crosstalk",
    "inaudible",
    "indistinct",
    "indistinct chatter",
    "laughing",
    "laughs",
    "laughter",
    "no speech",
    "noise",
    "sighs",
    "silence",
    "static",
];

const MUSIC_NOTES: &[char] = &['♪', '♫', '♬', '♩'];

/// A segment whose words are mostly one repeated n-gram is a decoder loop.
const LOOP_MIN_WORDS: usize = 10;
const LOOP_MAX_NGRAM: usize = 4;
const LOOP_RATIO: f32 = 0.7;

/// Consecutive repeats collapsed to one copy; shorter runs are left alone so
/// ordinary emphasis ("no no no", "very, very") is untouched.
const MIN_WORD_REPEATS: usize = 8;
const MIN_PHRASE_REPEATS: usize = 4;
const MAX_PHRASE_WORDS: usize = 12;

fn is_separator(c: char) -> bool {
    if c.is_whitespace() || c.is_ascii_punctuation() || MUSIC_NOTES.contains(&c) {
        return true;
    }
    if c.is_alphanumeric() {
        return false;
    }
    matches!(c,
        '\u{2000}'..='\u{206F}' // general punctuation: dashes, quotes, ellipsis
        | '\u{3000}'..='\u{303F}' // CJK punctuation
        | '\u{FF01}'..='\u{FF0F}' | '\u{FF1A}'..='\u{FF20}' | '\u{FF3B}'..='\u{FF40}'
        | '\u{FF5B}'..='\u{FF65}' // full-width punctuation
        | '¡' | '¿' | '«' | '»' | '।' | '॥')
}

/// Byte spans of the words in `text`.
fn word_spans(text: &str) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut start = None;
    for (i, c) in text.char_indices() {
        match (is_separator(c), start) {
            (true, Some(s)) => {
                spans.push((s, i));
                start = None;
            }
            (false, None) => start = Some(i),
            _ => {}
        }
    }
    if let Some(s) = start {
        spans.push((s, text.len()));
    }
    spans
}

fn words(text: &str) -> Vec<String> {
    word_spans(text)
        .into_iter()
        .map(|(a, b)| text[a..b].to_lowercase())
        .collect()
}

fn is_non_speech_tag(inner: &str) -> bool {
    let w = words(inner);
    w.iter().any(|w| w == "music") || NON_SPEECH_TAGS.contains(&w.join(" ").as_str())
}

/// Remove `[...]` annotations, known `(...)` non-speech annotations, and music
/// note symbols.
fn strip_tags(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(open) = rest.find(['[', '(', '（']) {
        let opener = rest[open..].chars().next().unwrap();
        let closer = match opener {
            '[' => ']',
            '(' => ')',
            _ => '）',
        };
        let after = open + opener.len_utf8();
        match rest[after..].find(closer) {
            Some(len) if !rest[after..after + len].contains(opener) => {
                let end = after + len + closer.len_utf8();
                if opener == '[' || is_non_speech_tag(&rest[after..after + len]) {
                    out.push_str(&rest[..open]);
                } else {
                    out.push_str(&rest[..end]);
                }
                rest = &rest[end..];
            }
            _ => {
                out.push_str(&rest[..after]);
                rest = &rest[after..];
            }
        }
    }
    out.push_str(rest);
    out.retain(|c| !MUSIC_NOTES.contains(&c));
    out
}

fn is_stock_phrase(text: &str, words: &[String], no_speech_probability: f32) -> bool {
    if text.to_lowercase().contains("amara.org") {
        return true;
    }
    if no_speech_probability < NO_SPEECH_PROBABILITY_THRESHOLD {
        return false;
    }
    let joined = words.join(" ");
    SILENCE_PHRASES.contains(&joined.as_str())
        || CREDIT_PREFIXES
            .iter()
            .any(|p| joined == *p || joined.starts_with(&format!("{p} ")))
}

/// Share of the words covered by the most frequent n-gram (non-overlapping
/// occurrences), for n up to `LOOP_MAX_NGRAM`.
fn repetition_ratio(words: &[String]) -> f32 {
    let mut best = 0.0f32;
    for n in 1..=LOOP_MAX_NGRAM.min(words.len()) {
        let mut seen: HashMap<&[String], (usize, usize)> = HashMap::new();
        for i in 0..=words.len() - n {
            let entry = seen.entry(&words[i..i + n]).or_insert((0, 0));
            if i >= entry.1 {
                entry.0 += 1;
                entry.1 = i + n;
            }
        }
        let top = seen.values().map(|(count, _)| *count).max().unwrap_or(0);
        best = best.max((top * n) as f32 / words.len() as f32);
    }
    best
}

fn is_loop(words: &[String]) -> bool {
    words.len() >= LOOP_MIN_WORDS && repetition_ratio(words) > LOOP_RATIO
}

/// Collapse long consecutive runs of the same word or phrase to one copy,
/// leaving the rest of `text` byte-for-byte intact.
pub fn collapse_repetitions(text: &str) -> String {
    let spans = word_spans(text);
    let keys: Vec<String> = spans
        .iter()
        .map(|&(a, b)| text[a..b].to_lowercase())
        .collect();
    let mut out = String::with_capacity(text.len());
    let mut cursor = 0;
    let mut i = 0;
    while i < keys.len() {
        let run = (1..=MAX_PHRASE_WORDS).find_map(|n| {
            let mut reps = 1;
            while i + (reps + 1) * n <= keys.len()
                && keys[i + reps * n..i + (reps + 1) * n] == keys[i..i + n]
            {
                reps += 1;
            }
            let min = if n == 1 {
                MIN_WORD_REPEATS
            } else {
                MIN_PHRASE_REPEATS
            };
            (reps >= min).then_some((n, reps))
        });
        match run {
            Some((n, reps)) => {
                out.push_str(&text[cursor..spans[i + n - 1].1]);
                cursor = spans[i + reps * n - 1].1;
                i += reps * n;
            }
            None => i += 1,
        }
    }
    out.push_str(&text[cursor..]);
    out
}

/// Clean one Whisper segment. Returns an empty string for a segment that is
/// a silence hallucination or a decoder loop.
pub fn sanitize_segment(text: &str, no_speech_probability: f32) -> String {
    let cleaned = strip_tags(text);
    let words = words(&cleaned);
    if words.is_empty() {
        return cleaned.chars().filter(|c| c.is_whitespace()).collect();
    }
    if is_loop(&words) || is_stock_phrase(&cleaned, &words, no_speech_probability) {
        return String::new();
    }
    collapse_repetitions(&cleaned)
}

/// Clean output from a model that gives no no-speech probability (Parakeet).
/// Its stock phrases are never treated as silence: Parakeet was not trained on
/// subtitles and does not invent them, so "Thank you." is speech. Tags and
/// decoder loops are removed as for Whisper.
pub fn sanitize_unscored(text: &str) -> String {
    sanitize_segment(text, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parakeet_output_keeps_short_phrases_and_fillers() {
        for text in [
            "Thank you.",
            "Bye.",
            "Uh, so, um, I think we should, uh, ship it on Friday.",
            "Twenty twenty six was the year we moved to three hundred and fifty users.",
            "Yeah. Yeah. Yeah.",
            "Mm-hmm.",
        ] {
            assert_eq!(sanitize_unscored(text), text, "{text}");
        }
    }

    #[test]
    fn parakeet_loops_on_vocal_noise_are_dropped_or_collapsed() {
        // A whole piece that is one looping n-gram (humming, a laugh track).
        assert_eq!(sanitize_unscored(&"la la ".repeat(10)), "");
        assert_eq!(sanitize_unscored(&"Oh, oh, oh, oh, ".repeat(4)), "");
        // A loop after real speech keeps the speech and one copy of the loop.
        let looped = format!("We should test this before the release on Friday.{}", " Mm.".repeat(12));
        assert_eq!(sanitize_unscored(&looped), "We should test this before the release on Friday. Mm.");
        assert_eq!(sanitize_unscored("Subtitles by the Amara.org community"), "");
    }

    #[test]
    fn strips_blank_audio_markers() {
        assert_eq!(sanitize_segment(" [BLANK_AUDIO] ", 0.0), "  ");
        assert_eq!(
            sanitize_segment("hello [blank_audio] world", 0.0),
            "hello  world"
        );
    }

    #[test]
    fn drops_thank_you_only_when_model_detects_no_speech() {
        assert_eq!(sanitize_segment("Thank you.", 0.9), "");
        assert_eq!(sanitize_segment("Thank you.", 0.1), "Thank you.");
    }

    #[test]
    fn drops_stock_silence_phrases_only_when_model_detects_no_speech() {
        for phrase in [
            " you",
            " Thanks for watching!",
            " Thank you for watching.",
            " Subtitles by the Amara.org community",
            " Transcription by CastingWords",
            "ご視聴ありがとうございました。",
            " Продолжение следует...",
        ] {
            assert_eq!(sanitize_segment(phrase, 0.8), "", "{phrase}");
        }
        for phrase in [
            " you",
            " Thanks for watching!",
            " Transcription by CastingWords",
        ] {
            assert_eq!(sanitize_segment(phrase, 0.1), phrase, "{phrase}");
        }
        // Spoken inside a longer sentence, the phrase is speech.
        assert_eq!(
            sanitize_segment(" Thank you for joining, everyone.", 0.9),
            " Thank you for joining, everyone."
        );
        // Amara credits are never speech.
        assert_eq!(sanitize_segment("字幕由Amara.org社区提供", 0.0), "");
    }

    #[test]
    fn removes_non_speech_tags_unconditionally() {
        for tag in [
            "[Music]",
            " (music)",
            "[silence]",
            " ♪",
            "♪♪♪",
            " [MUSIC PLAYING]",
            " (upbeat music)",
            "[音楽]",
            " (Applause)",
        ] {
            assert_eq!(sanitize_segment(tag, 0.0).trim(), "", "{tag}");
        }
        assert_eq!(
            sanitize_segment(" So that's the plan. [Music]", 0.0),
            " So that's the plan. "
        );
        assert_eq!(sanitize_segment(" ♪ la la la ♪", 0.0), "  la la la ");
        // Ordinary parentheses are kept.
        assert_eq!(
            sanitize_segment(" We ship Friday (maybe Monday).", 0.0),
            " We ship Friday (maybe Monday)."
        );
    }

    #[test]
    fn collapses_long_word_and_phrase_loops() {
        let loop_words = format!(" Okay so{}", " you".repeat(20));
        assert_eq!(collapse_repetitions(&loop_words), " Okay so you");
        let phrase = format!("{} and then we left.", " I went to the store.".repeat(6));
        assert_eq!(
            collapse_repetitions(&phrase),
            " I went to the store. and then we left."
        );
    }

    #[test]
    fn keeps_ordinary_repetition() {
        for text in [
            " No no no, that's not it.",
            " It was very, very good.",
            " I'm sorry, I'm sorry, I'm sorry.",
            " Ha ha ha ha.",
            " Hello hello, can you hear me? Hello?",
        ] {
            assert_eq!(sanitize_segment(text, 0.0), text, "{text}");
        }
    }

    #[test]
    fn drops_segments_that_are_mostly_one_ngram() {
        assert_eq!(
            sanitize_segment(
                " yeah okay yeah yeah okay yeah yeah yeah yeah yeah yeah yeah",
                0.0
            ),
            ""
        );
        assert_eq!(sanitize_segment(&" and the next one".repeat(5), 0.0), "");
        let normal = " We agreed that the next release ships on Friday after the review.";
        assert_eq!(sanitize_segment(normal, 0.0), normal);
    }

    #[test]
    fn handles_non_latin_scripts() {
        // Devanagari words include combining marks (virama, vowel signs).
        let hindi = format!(" ठीक है{} चलो।", " मैं ठीक हूँ।".repeat(5));
        assert_eq!(collapse_repetitions(&hindi), " ठीक है मैं ठीक हूँ। चलो।");
        let emphasis = " बहुत बहुत धन्यवाद, आप सब का।";
        assert_eq!(sanitize_segment(emphasis, 0.0), emphasis);
        assert_eq!(sanitize_segment(&" नमस्ते".repeat(12), 0.0), "");
        assert_eq!(words(" मैं ठीक हूँ।"), ["मैं", "ठीक", "हूँ"]);
    }
}
