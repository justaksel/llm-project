import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import math
import gradio as gr

# Modeli yükle (Zaten inmiş olan modeli kullanıyoruz, vakit kaybetmeyeceğiz)
model_id = "gpt2" # Eğer distilgpt2 indirdiysen burayı "distilgpt2" yap
tokenizer = GPT2Tokenizer.from_pretrained(model_id)
model = GPT2LMHeadModel.from_pretrained(model_id)

def bot_cevapla(soru):
    # Modeli bir bot gibi davranmaya zorlayan Soru-Cevap formatı
    prompt = f"Question: {soru}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Botun cevap üretmesi
    outputs = model.generate(
        inputs["input_ids"], 
        max_new_tokens=100, # Sadece yeni üretilen kelime sayısı
        num_return_sequences=1, 
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=2 # Tekrara düşmesini engeller
    )
    
    tam_metin = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Ekrana sadece "Answer:" kısmından sonrasını (Botun cevabını) yazdırmak için
    try:
        cevap = tam_metin.split("Answer:")[1].strip()
    except:
        cevap = tam_metin
        
    # Perplexity (Rapor metrikleri için)
    loss = model(inputs["input_ids"], labels=inputs["input_ids"]).loss
    perplexity = math.exp(loss.item())
    
    return cevap, f"Perplexity Skoru: {perplexity:.2f}"

# Arayüzü Chatbot formatına çevir
arayuz = gr.Interface(
    fn=bot_cevapla,
    inputs=gr.Textbox(lines=2, placeholder="İngilizce bir soru sorun (Örn: What is the capital of France?)..."),
    outputs=[gr.Textbox(label="Botun Cevabı"), gr.Textbox(label="Değerlendirme Metriği")],
    title="Intelligent QA Bot",
    description="Bu sohbet aracısı (conversational agent), GPT mimarisini kullanarak sorduğunuz sorulara yanıt üretir."
)

arayuz.launch()