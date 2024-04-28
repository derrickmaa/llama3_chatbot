import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig, BitsAndBytesConfig
import torch
from messageformat import MessageFormat


@st.cache_resource
def load_model():
    model_id = "/mnt/f/llama3/huggingface/Meta-Llama-3-8B"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config
    )
    return tokenizer, model

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
)

tokenizer, model = load_model()

generation_config = GenerationConfig(
    do_sample=True,
    top_p=0.9,
    temperature=0.6,
    max_new_tokens=1024
)

def main():
    st.title("💬 Chatbot")
    if st.button("clear history"):
        st.session_state["messages"] = []
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        for message in st.session_state.messages:
            st.chat_message(message['role']).write(message['content'])
        format_input = MessageFormat(tokenizer)
        input_messages = format_input.encode_dialog_prompt(st.session_state["messages"])
        input_tokens = torch.tensor(input_messages, dtype=torch.long, device="cuda").view(1,-1)
        generated_sequence = model.generate(
            input_ids = input_tokens,
            pad_token_id=tokenizer.convert_tokens_to_ids("<|end_of_text|>"),
            generation_config=generation_config
        )
        generated_sequence = generated_sequence.view(-1).tolist()
        generated_text = tokenizer.decode(
            generated_sequence,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        prompt_length = len(
                        tokenizer.decode(
                            input_tokens.view(-1),
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=True,
                        )
        )
        response = generated_text[prompt_length:]
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)

if __name__=="__main__":
    main()
