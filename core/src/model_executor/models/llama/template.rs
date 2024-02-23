use crate::model_executor::models::template::{ChatTemplate, JinjaTemplate};
use minijinja::context;

pub struct LlamaChatTemplate {
    template: JinjaTemplate,
}

impl LlamaChatTemplate {
    pub fn new(content: &str) -> anyhow::Result<Self> {
        let content = content.replace(".strip()", "|trim");
        //     let content = "{% if messages[0]['role'] == 'system' %}
        // {% set loop_messages = messages[1:] %}
        // {% set system_message = messages[0]['content'] %}
        // {% elif USE_DEFAULT_PROMPT == true and not '<<SYS>>' in messages[0]['content'] %}
        // {% set loop_messages = messages %}
        // {% set system_message = 'DEFAULT_SYSTEM_MESSAGE' %}
        // {% else %}
        // {% set loop_messages = messages %}
        // {% set system_message = false %}
        // {% endif %}
        // {% if loop_messages|length == 0 and system_message %}
        // {{ bos_token + '[INST] <SYS>\\n' + system_message + '\\n</SYS>\\n\\n [/INST]' }}
        // {% endif %}
        // {% for message in loop_messages %}
        // {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
        // {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
        // {% endif %}
        // {% if loop.index0 == 0 and system_message != false %}
        // {% set content = '<<SYS>>\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}
        // {% else %}
        // {% set content = message['content'] %}
        // {% endif %}
        // {% if message['role'] == 'user' %}
        // {{ bos_token + '[INST] ' + content+ ' [/INST]' }}
        // {% elif message['role'] == 'system' %}
        // {{ '<SYS>\\n' + content + '\\n</SYS>\\n\\n' }}
        // {% elif message['role'] == 'assistant' %}
        // {{ ' '  + content + ' ' + eos_token }}
        // {% endif %}
        // {% endfor %}";
        let t = JinjaTemplate::new(content.as_str())?;
        Ok(Self { template: t })
    }
}

impl ChatTemplate for LlamaChatTemplate {
    fn apply(
        &self,
        input: &Vec<std::collections::HashMap<String, String>>,
    ) -> anyhow::Result<String> {
        let ctx = context! {
            eos_token=>"</s>",
            bos_token=>"<s>",
            USE_DEFAULT_PROMPT=>false
        };
        let mut messages = Vec::new();
        for dict in input {
            if !dict.contains_key("role") || !dict.contains_key("content") {
                return Err(anyhow::anyhow!("no role/content in dict"));
            }
            let role = dict.get("role").unwrap();
            let content = dict.get("content").unwrap();
            let msg = context! {
                role,
                content,
            };
            messages.push(msg);
        }
        let ctx = context! { ..ctx, ..context! {
            messages
        }};
        self.template.apply(ctx)
    }
}
