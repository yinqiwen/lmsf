use minijinja::Environment;
use serde::Serialize;

pub struct ChatTemplate {
    _template: String,
    env: Environment<'static>,
}

impl ChatTemplate {
    pub fn new(content: &str) -> anyhow::Result<Self> {
        let content = content.replace(".strip()", "|trim");
        let mut env = Environment::new();
        let template_content = String::from(content);
        env.add_template_owned("default", template_content.clone())
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        Ok(Self {
            _template: template_content,
            env,
        })
    }
    pub fn apply<S: Serialize>(&self, ctx: S) -> anyhow::Result<String> {
        let template = self
            .env
            .get_template("default")
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        let val = template.render(ctx).map_err(|e| anyhow::anyhow!("{}", e))?;
        Ok(String::from(val.trim()))
    }
}

#[test]
fn test_chat_template() -> anyhow::Result<()> {
    use minijinja::context;
    let _DEFAULT_SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your \
    answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure\
     that your responses are socially unbiased and positive in nature.

    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not \
    correct. If you don't know the answer to a question, please don't share false information.";
    let _t2 = "{% for message in messages %}\
    {% if message['role'] == 'user' %}\
    {{ '<|user|>\n' + message['content'] + eos_token }}\
    {% elif message['role'] == 'system' %}\
    {{ '<|system|>\n' + message['content'] + eos_token }}\
    {% elif message['role'] == 'assistant' %}\
    {{ '<|assistant|>\n'  + message['content'] + eos_token }}\
    {% endif %}\
    {% if loop.last and add_generation_prompt %}\
    {{ '<|assistant|>' }}\
    {% endif %}\
    {% endfor %}";

    let t = "{% if messages[0]['role'] == 'system' %}
    {% set loop_messages = messages[1:] %}
    {% set system_message = messages[0]['content'] %}
    {% elif USE_DEFAULT_PROMPT == true and not '<<SYS>>' in messages[0]['content'] %}
    {% set loop_messages = messages %} 
    {% set system_message = 'DEFAULT_SYSTEM_MESSAGE' %}
    {% else %}
    {% set loop_messages = messages %}
    {% set system_message = false %}
    {% endif %}
    {% if loop_messages|length == 0 and system_message %}
    {{ bos_token + '[INST] <SYS>\\n' + system_message + '\\n</SYS>\\n\\n [/INST]' }}
    {% endif %}
    {% for message in loop_messages %}
    {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
    {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
    {% endif %}
    {% if loop.index0 == 0 and system_message != false %}
    {% set content = '<SYS>\\n' + system_message + '\\n</SYS>\\n\\n' + message['content'] %}
    {% else %}
    {% set content = message['content'] %}
    {% endif %}
    {% if message['role'] == 'user' %}
    {{ bos_token + '[INST] ' + content|trim + ' [/INST]' }}
    {% elif message['role'] == 'system' %}
    {{ '<SYS>\\n' + content + '\\n</SYS>\\n\\n' }}
    {% elif message['role'] == 'assistant' %}
    {{ ' '  + content|trim + ' ' + eos_token }}
    {% endif %}
    {% endfor %}";

    let t = ChatTemplate::new(t)?;
    let ctx = context! { messages =>vec![
        context!(role => "system", content => "test"),
        context!(role => "user", content => "hello"),
    ],
    eos_token=>"</s>",
    bos_token=>"<s>",
    USE_DEFAULT_PROMPT=>false};

    // let mut ctx = Vec::new();
    // let mut msg0 = HashMap::new();
    // msg0.insert("role".to_string(), "system".to_string());
    // msg0.insert("content".to_string(), "test".to_string());
    // let mut msg1 = HashMap::new();
    // msg1.insert("role".to_string(), "user".to_string());
    // msg1.insert("content".to_string(), "hello".to_string());
    // ctx.push(msg0);
    // ctx.push(msg1);
    let s = t.apply(ctx)?;
    println!("##{}##", s);

    // let template = env.get_template("hello").unwrap();
    // let ctx = context! { name => "xzd" };
    // println!("##{}##", template.render(ctx).unwrap());
    // let chat_template = ChatTemplate::new(t2);
    // let mut context = Context::new();

    // let s = chat_template.apply(&context).unwrap();
    Ok(())
}
