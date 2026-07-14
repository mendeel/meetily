use super::profiles::PolishProfile;

const FILLERS: &[&str] = &["um", "uh", "erm", "like"];

/// Trim, remove whole-token fillers, collapse whitespace, strip leading leftover commas.
pub fn rule_based_cleanup(text: &str) -> String {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    let mut tokens: Vec<String> = Vec::new();
    let mut current = String::new();

    for ch in trimmed.chars() {
        if ch.is_whitespace() {
            flush_token(&mut current, &mut tokens);
        } else if matches!(ch, ',' | '.' | ';' | ':' | '!' | '?') {
            flush_token(&mut current, &mut tokens);
            // Keep punctuation only when it isn't orphaned after filler removal;
            // attach as its own token and clean later.
            tokens.push(ch.to_string());
        } else {
            current.push(ch);
        }
    }
    flush_token(&mut current, &mut tokens);

    // Drop filler whole-tokens (case-insensitive exact match on word part).
    let mut kept: Vec<String> = Vec::new();
    for tok in tokens {
        if tok.len() == 1 && matches!(tok.chars().next(), Some(',' | '.' | ';' | ':' | '!' | '?')) {
            kept.push(tok);
            continue;
        }
        if is_filler(&tok) {
            continue;
        }
        kept.push(tok);
    }

    // Rebuild: spaces between words; punctuation sticks to previous word; strip leading commas.
    let mut out = String::new();
    for tok in kept {
        let is_punct = tok.len() == 1 && matches!(tok.chars().next(), Some(',' | '.' | ';' | ':' | '!' | '?'));
        if is_punct {
            if out.is_empty() {
                // strip leading leftover commas (and other leading punct)
                continue;
            }
            out.push_str(&tok);
        } else {
            if !out.is_empty() {
                let last = out.chars().last().unwrap();
                if !matches!(last, ',' | '.' | ';' | ':' | '!' | '?') {
                    out.push(' ');
                } else {
                    out.push(' ');
                }
            }
            out.push_str(&tok);
        }
    }

    // Collapse any leftover multi-spaces and trim leading commas/spaces again.
    let collapsed: String = out.split_whitespace().collect::<Vec<_>>().join(" ");
    collapsed
        .trim_start_matches(|c: char| c == ',' || c.is_whitespace())
        .trim()
        .to_string()
}

fn flush_token(current: &mut String, tokens: &mut Vec<String>) {
    if !current.is_empty() {
        tokens.push(std::mem::take(current));
    }
}

fn is_filler(tok: &str) -> bool {
    let lower = tok.to_lowercase();
    FILLERS.iter().any(|f| *f == lower)
}

pub fn system_prompt_for(profile: PolishProfile) -> &'static str {
    match profile {
        PolishProfile::Ide => {
            "You lightly clean up dictated technical text for an IDE. Keep code identifiers, \
             paths, and symbols intact. Fix only obvious speech artifacts. No fluff."
        }
        PolishProfile::Chat => {
            "You polish dictated chat messages: fix grammar, keep a natural tone, \
             and return a clear message ready to send."
        }
        PolishProfile::Email => {
            "You polish dictated email into clear, professional prose with correct grammar \
             and sensible paragraphs. Keep the original meaning."
        }
        PolishProfile::Default => {
            "You apply moderate cleanup to dictated text: fix grammar lightly, \
             remove speech fillers, and preserve the speaker's meaning."
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_filler_and_collapses_space() {
        let out = rule_based_cleanup("  um, hello   uh world  ");
        assert_eq!(out, "hello world");
    }

    #[test]
    fn empty_stays_empty() {
        assert_eq!(rule_based_cleanup("   "), "");
        assert_eq!(rule_based_cleanup("um"), "");
    }

    #[test]
    fn ide_prompt_mentions_technical() {
        let p = system_prompt_for(PolishProfile::Ide);
        assert!(p.to_lowercase().contains("technical") || p.to_lowercase().contains("code"));
    }

    #[test]
    fn chat_prompt_mentions_grammar_or_message() {
        let p = system_prompt_for(PolishProfile::Chat);
        assert!(p.to_lowercase().contains("grammar") || p.to_lowercase().contains("message"));
    }
}
