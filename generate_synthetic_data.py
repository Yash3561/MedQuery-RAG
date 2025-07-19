# generate_synthetic_data.py (v4.3 - Polished with GenerationConfig)
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import time

# --- Load the LLM and Tokenizer ---
print("Loading Llama 3 8B for final dataset generation...")
model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Define the special tokens Llama 3 uses
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

# --- NEW: Define Generation Configurations ---
# Config for generating deterministic lists (no sampling)
list_gen_config = GenerationConfig(
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=False
)

# Config for generating creative, detailed text (with sampling)
detail_gen_config = GenerationConfig(
    max_new_tokens=1024,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.4,
    top_p=0.9
)

# --- The Expanded Category List ---
categories = [
    "Common Chronic Diseases (e.g., Hypertension)", "Acute Infectious Diseases (e.g., Influenza)",
    "Cardiovascular Conditions (e.g., Myocardial Infarction)", "Neurological Disorders (e.g., Parkinson's Disease)",
    "Degenerative Neurological Diseases", "Autoimmune Diseases (e.g., Rheumatoid Arthritis)",
    "Common Cancers (e.g., Lung Cancer)", "Rare Cancers",
    "Mental Health Disorders (e.g., Schizophrenia)", "Mood and Anxiety Disorders",
    "Dermatological Conditions (e.g., Eczema)", "Genetic Disorders (e.g., Sickle Cell Anemia)",
    "Rare Genetic Syndromes", "Endocrine and Metabolic Disorders (e.g., Hypothyroidism)",
    "Gastrointestinal and Digestive Diseases", "Liver and Biliary System Diseases",
    "Musculoskeletal and Bone Disorders", "Respiratory System Diseases",
    "Kidney and Urinary System Diseases", "Hematologic (Blood) Disorders",
    "Ophthalmologic (Eye) Conditions", "Otolaryngologic (Ear, Nose, Throat) Conditions",
    "Pediatric-Specific Illnesses", "Geriatric Health Syndromes", "Women's Health and Gynecological Issues"
]

# --- Batched Generation Logic with GenerationConfig ---
list_prompt_template = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>
You are a medical textbook author. Generate a comma-separated list of 10 distinct medical conditions, procedures, or syndromes that fall under the category of '{category}'.
Do not add any other text or explanation, just the comma-separated list.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

master_condition_list = []
print("Starting COMPREHENSIVE batched generation with professional config...")

for category in categories:
    print(f"--- Generating for category: {category} ---")
    prompt = list_prompt_template.format(category=category)
    inputs = tokenizer(prompt, return_tensors="pt").to(llm.device)
    
    # Use the specific config for list generation
    outputs = llm.generate(**inputs, generation_config=list_gen_config)
    
    condition_list_str = tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    if not condition_list_str:
        print(f"  > WARNING: Generated an empty list for category '{category}'. Skipping.")
        continue

    conditions_in_batch = [c.strip() for c in condition_list_str.split(',') if c.strip()]
    print(f"  > Got {len(conditions_in_batch)} items.")
    master_condition_list.extend(conditions_in_batch)
    time.sleep(1)

# --- De-duplication Logic ---
print("\n--- De-duplicating final list ---")
print(f"Total conditions generated (with potential duplicates): {len(master_condition_list)}")
unique_conditions = list(dict.fromkeys(master_condition_list))
print(f"Total unique conditions after cleanup: {len(unique_conditions)}")
conditions = unique_conditions

# --- Detail Generation Logic with GenerationConfig ---
detail_prompt_template = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>
You are a medical textbook author. For the medical topic '{condition}', write a detailed clinical summary. The summary must include these three sections, clearly marked: 'Symptoms:', 'Diagnosis:', and 'Treatment:'. If a section is not applicable (e.g., 'Symptoms' for a procedure), state 'Not Applicable'. Write detailed, realistic, and informative text for each section.

TOPIC: {condition}
CLINICAL SUMMARY:<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

generated_docs = []
for i, condition in enumerate(conditions):
    print(f"Generating details for: '{condition}' ({i+1}/{len(conditions)})")
    prompt = detail_prompt_template.format(condition=condition)
    inputs = tokenizer(prompt, return_tensors="pt").to(llm.device)
    
    # Use the specific config for detail generation
    outputs = llm.generate(**inputs, generation_config=detail_gen_config)
    
    generated_text = tokenizer.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    if not generated_text:
        print(f"  > WARNING: Generated empty details for '{condition}'. Skipping.")
        continue
        
    generated_docs.append({"condition": condition, "text": generated_text})

# --- Save to a CSV ---
df = pd.DataFrame(generated_docs)
save_path = "./data/synthetic_medical_notes.csv"
df.to_csv(save_path, index=False)

print(f"\nSUCCESS: 'Stunned & Impressed' dataset with {len(df)} unique records saved to {save_path}")