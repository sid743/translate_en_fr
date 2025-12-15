import customtkinter as ctk
import threading
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# ================= CONFIGURATION =================
# Path to your trained model folder
MODEL_PATH = "./mt_en_fr_t5_final" 
APP_TITLE = "NeuralSeq T5 Translator"
# =================================================

class TranslatorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # 1. Window Setup
        self.title(APP_TITLE)
        self.geometry("900x600")
        ctk.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
        ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

        # 2. Grid Layout Configuration
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=0) # Header
        self.grid_rowconfigure(1, weight=1) # Content
        self.grid_rowconfigure(2, weight=0) # Status

        # 3. Model Loading State
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 4. Build UI Components
        self.create_header()
        self.create_main_area()
        self.create_status_bar()

        # 5. Load Model in Background (so UI opens immediately)
        self.status_label.configure(text="⏳ Initializing Model... (this may take a moment)")
        threading.Thread(target=self.load_model, daemon=True).start()

    def create_header(self):
        header_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        header_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 10))
        
        title = ctk.CTkLabel(
            header_frame, 
            text=APP_TITLE, 
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title.pack(side="left")

        # Direction Switcher (Segmented Button looks professional)
        self.direction_var = ctk.StringVar(value="English → French")
        self.dir_switch = ctk.CTkSegmentedButton(
            header_frame,
            values=["English → French", "French → English"],
            variable=self.direction_var,
            font=ctk.CTkFont(size=14)
        )
        self.dir_switch.pack(side="right")

    def create_main_area(self):
        main_frame = ctk.CTkFrame(self, fg_color="transparent")
        main_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)
        main_frame.grid_columnconfigure(0, weight=1) # Input
        main_frame.grid_columnconfigure(1, weight=0) # Spacing
        main_frame.grid_columnconfigure(2, weight=1) # Output
        main_frame.grid_rowconfigure(0, weight=1)

        # --- LEFT SIDE (INPUT) ---
        input_container = ctk.CTkFrame(main_frame)
        input_container.grid(row=0, column=0, sticky="nsew")
        input_container.grid_rowconfigure(1, weight=1)
        input_container.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(input_container, text="Input Text", font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, sticky="w", padx=10, pady=10)
        
        self.input_textbox = ctk.CTkTextbox(
            input_container, 
            font=ctk.CTkFont(size=16),
            wrap="word"
        )
        self.input_textbox.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

        # --- RIGHT SIDE (OUTPUT) ---
        output_container = ctk.CTkFrame(main_frame)
        output_container.grid(row=0, column=2, sticky="nsew")
        output_container.grid_rowconfigure(1, weight=1)
        output_container.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(output_container, text="Translation", font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, sticky="w", padx=10, pady=10)

        self.output_textbox = ctk.CTkTextbox(
            output_container, 
            font=ctk.CTkFont(size=16),
            wrap="word",
            fg_color=("gray90", "gray20") # Slightly different color to indicate output
        )
        self.output_textbox.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self.output_textbox.configure(state="disabled") # Make read-only initially

        # --- CENTER BUTTON ---
        # A floating action button effect at the bottom center
        self.translate_btn = ctk.CTkButton(
            self,
            text="TRANSLATE",
            font=ctk.CTkFont(size=16, weight="bold"),
            height=50,
            command=self.start_translation,
            state="disabled" # Disabled until model loads
        )
        self.translate_btn.grid(row=1, column=0, sticky="s", pady=30)

    def create_status_bar(self):
        self.status_label = ctk.CTkLabel(
            self, 
            text="Status: Starting up...", 
            text_color="gray", 
            anchor="w"
        )
        self.status_label.grid(row=2, column=0, sticky="ew", padx=20, pady=10)

    def load_model(self):
        """Loads model in a background thread to prevent UI freezing"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
            self.model.to(self.device)
            
            # Update UI on main thread
            self.translate_btn.configure(state="normal")
            self.status_label.configure(text=f"✅ Model loaded successfully on {self.device.upper()}")
        except Exception as e:
            self.status_label.configure(text=f"❌ Error loading model: {str(e)}", text_color="red")
            print(e)

    def start_translation(self):
        """Wrapper to run inference in thread"""
        text = self.input_textbox.get("1.0", "end").strip()
        direction = self.direction_var.get()
        
        if not text:
            return

        self.status_label.configure(text="Running inference...")
        self.translate_btn.configure(state="disabled")
        
        # Run inference in separate thread
        threading.Thread(target=self.run_inference, args=(text, direction)).start()

    def run_inference(self, text, direction):
        try:
            # Prepare Prefix
            if direction == "English → French":
                prefix = "translate English to French: "
            else:
                prefix = "translate French to English: "
            
            input_text = prefix + text
            
            # Tokenize
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)

            # Generate
            outputs = self.model.generate(
                **inputs,
                max_length=128,
                num_beams=4,
                length_penalty=0.8,
                early_stopping=True
            )
            
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Update UI
            self.update_output(result)
            self.status_label.configure(text="✅ Translation complete.")
            
        except Exception as e:
            self.status_label.configure(text=f"Error: {str(e)}")
        
        finally:
            self.translate_btn.configure(state="normal")

    def update_output(self, text):
        # Textboxes must be enabled to write, then disabled again to be read-only
        self.output_textbox.configure(state="normal")
        self.output_textbox.delete("1.0", "end")
        self.output_textbox.insert("1.0", text)
        self.output_textbox.configure(state="disabled")

if __name__ == "__main__":
    app = TranslatorApp()
    app.mainloop()