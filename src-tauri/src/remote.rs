//! Opt-in, bring-your-own-key remote processing for meetings.
//! Keys are loaded from the user's settings file; no key is bundled with Unsound.

use crate::audio::WHISPER_SAMPLE_RATE;
use crate::settings::{CloudProvider, Settings};
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE};
use serde_json::{json, Value};

fn selected<'a>(settings: &'a Settings, id: &str, purpose: &str) -> Result<&'a CloudProvider, String> {
    if id.trim().is_empty() {
        return Err(format!("choose a cloud {purpose} provider in Settings first"));
    }
    let provider = settings
        .cloud_providers
        .iter()
        .find(|p| p.id == id)
        .ok_or_else(|| format!("cloud provider '{id}' is not configured"))?;
    if provider.api_key.trim().is_empty() {
        return Err(format!("add an API key for {} in Settings", provider.id));
    }
    Ok(provider)
}

pub fn voice_ready(settings: &Settings) -> bool {
    selected(settings, &settings.cloud_voice_provider, "voice").is_ok()
}

pub fn text_ready(settings: &Settings) -> bool {
    selected(settings, &settings.cloud_text_provider, "text").is_ok()
}

fn wav(samples: &[f32]) -> Vec<u8> {
    let bytes_per_sample = 2u32;
    let data_len = samples.len() as u32 * bytes_per_sample;
    let mut out = Vec::with_capacity(44 + data_len as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&WHISPER_SAMPLE_RATE.to_le_bytes());
    out.extend_from_slice(&(WHISPER_SAMPLE_RATE * bytes_per_sample).to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for sample in samples {
        out.extend_from_slice(&((sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16).to_le_bytes());
    }
    out
}

async fn json_response(provider: &str, response: reqwest::Response) -> Result<Value, String> {
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(format!("{provider} request failed ({status}): {body}"));
    }
    serde_json::from_str(&body).map_err(|e| format!("{provider} returned invalid JSON: {e}"))
}

/// Transcribe a completed 16kHz mono segment. These APIs are deliberately
/// request-based first; Unsound's local VAD still provides natural chunks.
pub async fn transcribe(settings: &Settings, samples: &[f32], language: Option<&str>) -> Result<String, String> {
    let p = selected(settings, &settings.cloud_voice_provider, "voice")?;
    let model = p.voice_model.trim();
    if model.is_empty() {
        return Err(format!("choose a voice model for {} in Settings", p.id));
    }
    let audio = wav(samples);
    let client = reqwest::Client::new();
    let response = match p.id.as_str() {
        "openai" | "mistral" => {
            let url = if p.id == "openai" {
                "https://api.openai.com/v1/audio/transcriptions"
            } else {
                "https://api.mistral.ai/v1/audio/transcriptions"
            };
            let mut form = reqwest::multipart::Form::new()
                .text("model", model.to_string())
                .part(
                    "file",
                    reqwest::multipart::Part::bytes(audio)
                        .file_name("segment.wav")
                        .mime_str("audio/wav")
                        .map_err(|e| e.to_string())?,
                );
            if let Some(lang) = language.filter(|v| !v.is_empty()) {
                form = form.text("language", lang.to_string());
            }
            client
                .post(url)
                .header(AUTHORIZATION, format!("Bearer {}", p.api_key))
                .multipart(form)
                .send()
                .await
                .map_err(|e| e.to_string())?
        }
        "deepgram" => client
            .post(format!("https://api.deepgram.com/v1/listen?model={model}&smart_format=true"))
            .header(AUTHORIZATION, format!("Token {}", p.api_key))
            .header(CONTENT_TYPE, "audio/wav")
            .body(audio)
            .send()
            .await
            .map_err(|e| e.to_string())?,
        "elevenlabs" => {
            let mut form = reqwest::multipart::Form::new()
                .text("model_id", model.to_string())
                .part(
                    "file",
                    reqwest::multipart::Part::bytes(audio)
                        .file_name("segment.wav")
                        .mime_str("audio/wav")
                        .map_err(|e| e.to_string())?,
                );
            if let Some(lang) = language.filter(|v| !v.is_empty()) {
                form = form.text("language_code", lang.to_string());
            }
            client
                .post("https://api.elevenlabs.io/v1/speech-to-text")
                .header("xi-api-key", &p.api_key)
                .multipart(form)
                .send()
                .await
                .map_err(|e| e.to_string())?
        }
        other => return Err(format!("unsupported cloud voice provider: {other}")),
    };
    let json = json_response(&p.id, response).await?;
    let text = match p.id.as_str() {
        "deepgram" => json
            .pointer("/results/channels/0/alternatives/0/transcript")
            .and_then(Value::as_str),
        _ => json.get("text").and_then(Value::as_str),
    }
    .unwrap_or_default()
    .trim()
    .to_string();
    Ok(text)
}

async fn chat(settings: &Settings, system: &str, user: &str) -> Result<String, String> {
    let p = selected(settings, &settings.cloud_text_provider, "text")?;
    let model = p.text_model.trim();
    if model.is_empty() {
        return Err(format!("choose a text model for {} in Settings", p.id));
    }
    let url = match p.id.as_str() {
        "openai" => "https://api.openai.com/v1/chat/completions",
        "mistral" => "https://api.mistral.ai/v1/chat/completions",
        other => return Err(format!("{other} is not available for meeting text; choose OpenAI or Mistral")),
    };
    let response = reqwest::Client::new()
        .post(url)
        .header(AUTHORIZATION, format!("Bearer {}", p.api_key))
        .json(&json!({
            "model": model,
            "temperature": 0.1,
            "messages": [{"role":"system","content":system},{"role":"user","content":user}]
        }))
        .send()
        .await
        .map_err(|e| e.to_string())?;
    let json = json_response(&p.id, response).await?;
    json
        .pointer("/choices/0/message/content")
        .and_then(Value::as_str)
        .map(str::trim)
        .map(str::to_string)
        .filter(|text| !text.is_empty())
        .ok_or_else(|| format!("{} returned no text", p.id))
}

pub async fn summarize(settings: &Settings, transcript: &str) -> Result<String, String> {
    chat(settings, crate::llm::MEETING_SUMMARY_PROMPT, &format!("Transcript:\n\n{transcript}")).await
}

pub async fn answer(settings: &Settings, context: &str, question: &str) -> Result<String, String> {
    chat(settings, crate::llm::MEETING_QA_PROMPT, &format!("{context}\n\nQuestion: {question}")).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_a_valid_pcm_wav_header() {
        let bytes = wav(&[0.0, 0.5]);
        assert_eq!(&bytes[..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        assert_eq!(bytes.len(), 48);
    }
}
