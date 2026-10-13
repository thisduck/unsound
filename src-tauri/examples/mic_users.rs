//! Print which processes are using the microphone right now, and the
//! calendar permission state — a quick check of `detect` / `calendar`
//! outside the app. Run: `cargo run --example mic_users`
fn main() {
    match unsound_lib::detect::mic_users() {
        Some(users) if users.is_empty() => println!("mic users: none"),
        Some(users) => {
            println!("mic users:");
            for u in users {
                println!("  pid {:>6}  {}", u.pid, if u.bundle_id.is_empty() { "(no bundle id)" } else { &u.bundle_id });
            }
        }
        None => println!("mic users: per-process API unavailable (needs macOS 14.2+)"),
    }
    println!("calendar access: {:?}", unsound_lib::calendar::access());
}
