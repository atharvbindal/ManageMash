import tkinter
import customtkinter
import time
import threading
from typing import Dict, Any, Callable

from obfuscation_detector import quick_flag_check, detailed_analysis
from translate import translate_text
from training.inference_script import load_finetuned_model_for_inference
from transformers import pipeline
from Pii_redactor import mask_pii
from MAINLLM import generate_text_with_groq
from translateBack import translate_back

BOX_FONT = ("Consolas", 12)
TITLE_FONT = ("Arial", 14, "bold")

jailbreak_classifier = None

def initialize_jailbreak_model_for_ui(update_callback: Callable):
    global jailbreak_classifier
    if jailbreak_classifier is None:
        try:
            update_callback("jailbreak", "🔧 Initializing model...")
            import os
            os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'
            MODEL_NAME = "distilbert-base-uncased"
            OUTPUT_MODEL_DIR = "C:\\Unified coding\\Projects\\Managemash\\training\\fine_tuned_slm_adapter_jailbreak"
            jailbreak_model, tokenizer = load_finetuned_model_for_inference(OUTPUT_MODEL_DIR, MODEL_NAME)
            jailbreak_classifier = pipeline("text-classification", model=jailbreak_model, tokenizer=tokenizer)
            return True
        except Exception as e:
            error_msg = f"❌ Failed to initialize jailbreak model: {e}"
            update_callback("jailbreak", error_msg)
            return False
    return True

def process_text_through_defense_for_ui(input_text: str, update_callback: Callable[[str, str], None]):
    global jailbreak_classifier

    def update_ui(step, message):
        update_callback(step, message)

    update_ui("obfuscation", "Running...")
    is_obfuscated = quick_flag_check(input_text)
    if is_obfuscated:
        analysis = detailed_analysis(input_text)
        output = f"🚩 BLOCKED: Obfuscation detected. Reasons: {', '.join(analysis['flag_reasons'])}"
        update_ui("obfuscation", output)
        update_ui("final_output", "🛑 Final Output: Sorry, flagged ")
        return
    else:
        update_ui("obfuscation", "✅ PASSED: No obfuscation detected")

    update_ui("translation", "Running...")
    original_language, translated_text = translate_text(input_text)
    stored_language, current_text = None, input_text
    if original_language and translated_text and original_language != 'en':
        output = f"🔄 Text translated from {original_language} to English"
        update_ui("translation", output)
        stored_language, current_text = original_language, translated_text
    else:
        update_ui("translation", "✅ Text is already in English or translation not needed")

    if not initialize_jailbreak_model_for_ui(update_callback):
        return
        
    update_ui("jailbreak", "Running...")
    
    result = jailbreak_classifier(current_text)
    if result[0]['label'] == "JAILBREAK" and result[0]['score'] > 0.86:
        output = f"🚩 BLOCKED: Jailbreak detected (Score: {result[0]['score']:.4f})"
        update_ui("jailbreak", output)
        update_ui("final_output", "🛑 Final Output: Sorry, flagged ")
        return
    else:
        output = f"✅ PASSED: No jailbreak detected (Score: {result[0]['score']:.4f})"
        update_ui("jailbreak", output)

    update_ui("pii1", "Running...")
    pii_redacted_text = mask_pii(current_text, aggregate_redaction=True)
    if pii_redacted_text != current_text:
        output = f"🔄 PII detected and redacted\n📝 Text after PII redaction: '{pii_redacted_text}'"
        update_ui("pii1", output)
        current_text = pii_redacted_text
    else:
        update_ui("pii1", "✅ No PII detected")

    update_ui("llm", "Running...")
    llm_response = generate_text_with_groq(current_text)
    output = f"📝 LLM Response: '{llm_response}'"
    update_ui("llm", output)
    current_text = llm_response

    update_ui("pii2", "Running...")
    final_pii_redacted_text = mask_pii(current_text, aggregate_redaction=True)
    if final_pii_redacted_text != current_text:
        output = f"🔄 PII detected and redacted in LLM response\n📝 Text after final PII redaction: '{final_pii_redacted_text}'"
        update_ui("pii2", output)
        current_text = final_pii_redacted_text
    else:
        update_ui("pii2", "✅ No PII detected in LLM response")

    update_ui("translate_back", "Running...")
    final_output = current_text
    if stored_language:
        final_output = translate_back(current_text, stored_language)
        output = f"🌐 Translating back to {stored_language}\n📝 Final translated output: '{final_output}'"
        update_ui("translate_back", output)
    else:
        update_ui("translate_back", "✅ No translation back needed")

    update_ui("final_output", f"🏁 Final Output: '{final_output}'")

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        customtkinter.set_appearance_mode("dark")
        self.title("LLM Defense Pipeline Visualizer (Scrollable)")
        self.geometry("1200x800")

        self.scroll_container = customtkinter.CTkScrollableFrame(self)
        self.scroll_container.pack(fill="both", expand=True)

        self.main_frame = customtkinter.CTkFrame(self.scroll_container, fg_color="transparent")
        self.main_frame.pack(fill="both", expand=True)
        self.main_frame.grid_columnconfigure((0, 1, 2, 3, 4), weight=1, minsize=240)
        self.main_frame.grid_rowconfigure((0, 1, 2, 3, 4), weight=1, minsize=180)

        self.canvas = tkinter.Canvas(self.main_frame, bg="#2b2b2b", highlightthickness=0)
        self.canvas.place(relx=0, rely=0, relwidth=1, relheight=1)

        self.steps = {}
        self.steps["user_input"] = self._create_box("User Input", 0, 1, is_input=True)
        self.steps["obfuscation"] = self._create_box("1. Obfuscation Detection", 1, 1)
        self.steps["translation"] = self._create_box("2. Translation", 2, 1)
        self.steps["jailbreak"] = self._create_box("3. Jailbreak Detection", 3, 1)
        self.steps["pii1"] = self._create_box("4. PII Redaction (In)", 4, 1)
        self.steps["llm"] = self._create_box("5. Main LLM", 4, 2)
        self.steps["pii2"] = self._create_box("6. PII Redaction (Out)", 2, 3)
        self.steps["translate_back"] = self._create_box("7. Translate Back", 1, 3)
        self.steps["final_output"] = self._create_box("Final Output", 0, 3)

        self.steps["user_input"]["textbox"].insert("0.0", "Hello, my name is John Doe. ")
        
        self.process_button = customtkinter.CTkButton(self.main_frame, text="Visualize Process", command=self.start_processing_thread)
        self.process_button.grid(row=2, column=0, pady=10)
        
        self.after(100, self.draw_arrows)
        self.bind("<Configure>", self.draw_arrows)

    def _create_box(self, title, col, row, is_input=False):
        frame = customtkinter.CTkFrame(self.main_frame, border_width=2, fg_color="#3c3c3c")
        frame.grid(row=row, column=col, padx=25, pady=40, sticky="nsew")
        
        title_label = customtkinter.CTkLabel(frame, text=title, font=TITLE_FONT)
        title_label.pack(pady=(5, 5), fill="x", padx=5)

        textbox = customtkinter.CTkTextbox(frame, font=BOX_FONT, wrap="word", fg_color="#2D2D2D")
        textbox.pack(expand=True, fill="both", padx=5, pady=5)
        
        if not is_input:
            textbox.insert("0.0", "Awaiting Input...")
            textbox.configure(state="disabled")
        
        return {"frame": frame, "textbox": textbox}

    def draw_arrows(self, event=None):
        self.canvas.delete("arrow")
        connections = [("user_input", "obfuscation"), ("obfuscation", "translation"), ("translation", "jailbreak"), 
                       ("jailbreak", "pii1"), ("pii1", "llm"), ("llm", "pii2"), ("pii2", "translate_back"), 
                       ("translate_back", "final_output")]
        for start, end in connections:
            self.main_frame.update_idletasks()
            f_frame = self.steps[start]["frame"]
            t_frame = self.steps[end]["frame"]
            fx, fy = f_frame.winfo_x() + f_frame.winfo_width()/2, f_frame.winfo_y() + f_frame.winfo_height()/2
            tx, ty = t_frame.winfo_x() + t_frame.winfo_width()/2, t_frame.winfo_y() + t_frame.winfo_height()/2
            self.canvas.create_line(fx, fy, tx, ty, fill="gray50", width=2, tags="arrow", arrow=tkinter.LAST)

    def update_step_box(self, step_key, text):
        if step_key in self.steps:
            textbox = self.steps[step_key]["textbox"]
            textbox.configure(state="normal")
            textbox.delete("1.0", "end")
            textbox.insert("0.0", text)
            textbox.configure(state="disabled")

    def reset_ui(self):
        for key in self.steps:
            if key != "user_input":
                    self.update_step_box(key, "Awaiting Input...")
        self.process_button.configure(state="normal", text="Visualize Process")

    def start_processing_thread(self):
        self.reset_ui()
        self.process_button.configure(state="disabled", text="Processing...")
        input_text = self.steps["user_input"]["textbox"].get("1.0", "end-1c")
        
        thread = threading.Thread(target=self.run_pipeline, args=(input_text,))
        thread.daemon = True
        thread.start()

    def run_pipeline(self, input_text):
        try:
            process_text_through_defense_for_ui(input_text, self.update_step_box)
        except Exception as e:
            self.update_step_box("final_output", f"An error occurred:\n{e}")
            import traceback
            traceback.print_exc()
        finally:
            self.after(0, self.process_button.configure, {"state": "normal", "text": "Visualize Process"})

if __name__ == "__main__":
    app = App()
    app.mainloop()