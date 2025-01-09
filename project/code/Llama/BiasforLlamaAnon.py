# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("NousResearch/Hermes-3-Llama-3.1-8B")
model = AutoModelForCausalLM.from_pretrained("NousResearch/Hermes-3-Llama-3.1-8B")

# Use a pipeline as a high-level helper
from transformers import pipeline

import pandas as pd

df = pd.read_csv('df_subset.csv',  encoding='utf-8')
print(df)

for index, row in df.iterrows():
    print(index)
    question = row['question']
    student_response = row['answer_text']
    prompt = "Score the following response from a student given in the format Question: <question> Student's response: <student's response>. Question: " + question + " Student's response:  " + student_response + ". Score the student on a scale of 0 to 4, only provide a score."
    messages = [
            {"role": "user", "content": prompt},
            ]
    pipe = pipeline("text-generation", model="NousResearch/Hermes-3-Llama-3.1-8B", tokenizer=tokenizer, max_new_tokens=300)
    pipe(messages)
    outputs = pipe(messages, max_new_tokens=300,)
    print(outputs)
    output_text = outputs[0]["generated_text"][-1]['content']
    print(output_text)
    df.loc[index, 'Prompt'] = prompt
    df.loc[index, 'Output'] = output_text

print('done')
df.to_csv('answersFSP.csv')
print('done 2')
