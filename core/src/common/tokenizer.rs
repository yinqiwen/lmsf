use tokenizers::Tokenizer;

pub struct DecodeTokenOptions {
    pub prefix_offset: usize,
    pub read_offset: usize,
    pub skip_special_tokens: bool,
    pub spaces_between_special_tokens: bool,
    pub first_decoding: bool,
}

impl Default for DecodeTokenOptions {
    fn default() -> Self {
        Self {
            prefix_offset: 0,
            read_offset: 0,
            skip_special_tokens: false,
            spaces_between_special_tokens: true,
            first_decoding: false,
        }
    }
}

fn is_special_token_id(tokenizer: &Tokenizer, id: u32) -> anyhow::Result<bool> {
    let v = tokenizer
        .decode(&[id], true)
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    Ok(v.len() == 0)
}

pub fn decode_token(
    tokenizer: &Tokenizer,
    all_input_ids: &[u32],
    options: DecodeTokenOptions,
) -> anyhow::Result<(String, String, usize, usize)> {
    let prefix_text = tokenizer
        .decode(
            &all_input_ids[options.prefix_offset..options.read_offset],
            options.skip_special_tokens,
        )
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    if options.first_decoding {
        let new_token_id = *all_input_ids.last().unwrap();
        let new_token = tokenizer
            .decode(&[new_token_id], options.skip_special_tokens)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        let all_input_ids_len = all_input_ids.len();
        let prev_tokens = tokenizer
            .decode(
                &all_input_ids[options.prefix_offset..all_input_ids_len - 1],
                options.skip_special_tokens,
            )
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        let read_offset = all_input_ids.len() - 1;
        Ok((prev_tokens, new_token, read_offset, all_input_ids.len()))
    } else {
        let new_text = tokenizer
            .decode(
                &all_input_ids[options.prefix_offset..],
                options.skip_special_tokens,
            )
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        if new_text.len() > prefix_text.len() && !new_text.ends_with("�") {
            let new_text_slice = &new_text[prefix_text.len()..];
            Ok((
                String::new(),
                String::from(new_text_slice),
                options.read_offset,
                all_input_ids.len(),
            ))
        } else {
            Ok((
                String::new(),
                String::new(),
                options.prefix_offset,
                options.read_offset,
            ))
        }
    }
}

pub fn detokenize_incrementally(
    tokenizer: &Tokenizer,
    all_input_ids: &[u32],
    prev_tokens: Option<&Vec<String>>,
    options: DecodeTokenOptions,
) -> anyhow::Result<String> {
    let new_token_id = *all_input_ids.last().unwrap();
    let mut prev_tokens_cache = Vec::new();
    let (mut output_tokens, new_tokens, prefix_offset, read_offset) =
        if let Some(_prev_tokens) = prev_tokens {
            let output_tokens = prev_tokens_cache.iter().collect::<Vec<_>>();
            let new_tokens = tokenizer
                .decode(&[new_token_id], options.skip_special_tokens)
                .map_err(|e| anyhow::anyhow!("{}", e))?;

            (
                output_tokens,
                new_tokens,
                options.prefix_offset,
                options.read_offset,
            )
        } else {
            for id in all_input_ids {
                let tokens = tokenizer
                    .decode(&[*id], options.skip_special_tokens)
                    .map_err(|e| anyhow::anyhow!("{}", e))?;
                if !tokens.is_empty() {
                    prev_tokens_cache.push(tokens);
                }
            }
            let output_tokens = prev_tokens_cache.iter().collect::<Vec<_>>();
            let _new_tokens = tokenizer
                .decode(all_input_ids, options.skip_special_tokens)
                .map_err(|e| anyhow::anyhow!("{}", e))?;
            let prefix_offset = if output_tokens.len() > 6 {
                output_tokens.len() - 6
            } else {
                0
            };

            let read_offset =
                if options.skip_special_tokens && is_special_token_id(tokenizer, new_token_id)? {
                    0
                } else {
                    if output_tokens.len() > 1 {
                        output_tokens.len() - 1
                    } else {
                        0
                    }
                };
            (output_tokens, String::new(), prefix_offset, read_offset)
        };
    if !new_tokens.is_empty() {
        output_tokens.push(&new_tokens);
    }
    tracing::info!(
        "all_input_ids:{:?},output_tokens:{:?},new_tokens:{}",
        all_input_ids,
        output_tokens,
        new_tokens
    );
    let _prefix_text_tokens = &output_tokens[prefix_offset..read_offset];
    let _new_text_tokens = &output_tokens[prefix_offset..];
    let mut new_text: String = String::new();
    let prefix_text: String = String::new();
    //tokenizer.get_decoder().as_ref().unwrap().decode(tokens)

    // let prefix_text = output_tokens[prefix_offset..read_offset];
    // tokenizer.encode(input, add_special_tokens)

    // prefix_text = tokenizer.convert_tokens_to_string(
    //     output_tokens[prefix_offset:read_offset])
    // new_text = tokenizer.convert_tokens_to_string(
    //     output_tokens[prefix_offset:])
    if new_text.len() > prefix_text.len() && new_text.ends_with("�") {
        new_text = String::from(&new_text[prefix_text.len()..]);
    } else {
    }

    todo!("detokenize_incrementally")
}
