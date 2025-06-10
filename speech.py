#!/usr/bin/env python3
"""
🚀 Enhanced Multi-Language Live Speech Recognition with Whisper
Real-time transcription with proper auto language detection
Supports 99+ languages with accurate detection
"""

import torch
import numpy as np
import sounddevice as sd
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import warnings
import time
import re

warnings.filterwarnings("ignore")


class MultiLanguageSpeechRecognizer:
    def __init__(self, model_name="openai/whisper-tiny"):
        """Initialize the Whisper model for multi-language speech recognition"""
        print(f"🤖 Loading Whisper model: {model_name}")

        # Determine device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"💻 Using device: {self.device}")

        # Load model and processor
        print("📥 Loading model components...")
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Audio settings
        self.sample_rate = 16000
        self.chunk_duration = 4  # Increased for better language detection
        self.silence_threshold = 0.01

        # Language mapping for better display
        self.language_names = {
            'en': 'English', 'hi': 'Hindi', 'ur': 'Urdu', 'bn': 'Bengali', 'te': 'Telugu',
            'ta': 'Tamil', 'mr': 'Marathi', 'gu': 'Gujarati', 'kn': 'Kannada', 'ml': 'Malayalam',
            'pa': 'Punjabi', 'or': 'Odia', 'as': 'Assamese', 'ne': 'Nepali', 'si': 'Sinhala',
            'fr': 'French', 'es': 'Spanish', 'de': 'German', 'it': 'Italian', 'pt': 'Portuguese',
            'ru': 'Russian', 'ja': 'Japanese', 'ko': 'Korean', 'zh': 'Chinese', 'ar': 'Arabic',
            'tr': 'Turkish', 'pl': 'Polish', 'nl': 'Dutch', 'sv': 'Swedish', 'da': 'Danish',
            'no': 'Norwegian', 'fi': 'Finnish', 'cs': 'Czech', 'sk': 'Slovak', 'hu': 'Hungarian',
            'ro': 'Romanian', 'bg': 'Bulgarian', 'hr': 'Croatian', 'sr': 'Serbian', 'sl': 'Slovenian',
            'et': 'Estonian', 'lv': 'Latvian', 'lt': 'Lithuanian', 'uk': 'Ukrainian', 'be': 'Belarusian',
            'ka': 'Georgian', 'am': 'Amharic', 'sw': 'Swahili', 'yo': 'Yoruba', 'zu': 'Zulu',
            'af': 'Afrikaans', 'sq': 'Albanian', 'az': 'Azerbaijani', 'eu': 'Basque', 'bs': 'Bosnian',
            'ca': 'Catalan', 'cy': 'Welsh', 'el': 'Greek', 'eo': 'Esperanto', 'fa': 'Persian',
            'ga': 'Irish', 'gl': 'Galician', 'he': 'Hebrew', 'is': 'Icelandic', 'id': 'Indonesian',
            'jw': 'Javanese', 'kk': 'Kazakh', 'ky': 'Kyrgyz', 'la': 'Latin', 'lb': 'Luxembourgish',
            'mk': 'Macedonian', 'mg': 'Malagasy', 'ms': 'Malay', 'mt': 'Maltese', 'mi': 'Maori',
            'mn': 'Mongolian', 'my': 'Myanmar', 'ps': 'Pashto', 'sa': 'Sanskrit', 'gd': 'Scottish Gaelic',
            'sn': 'Shona', 'so': 'Somali', 'su': 'Sundanese', 'tg': 'Tajik', 'th': 'Thai',
            'tk': 'Turkmen', 'uz': 'Uzbek', 'vi': 'Vietnamese', 'xh': 'Xhosa', 'yi': 'Yiddish'
        }

        print("✅ Multi-language transcription model loaded successfully!")
        print(f"🌐 Supports {len(self.language_names)} languages")

    def is_speech_detected(self, audio_data, threshold=None):
        """Enhanced voice activity detection"""
        if threshold is None:
            threshold = self.silence_threshold

        # Calculate RMS energy
        rms = np.sqrt(np.mean(audio_data ** 2))

        # Calculate zero crossing rate
        zcr = np.mean(np.abs(np.diff(np.sign(audio_data))))

        # Simple spectral energy
        fft_data = np.abs(np.fft.fft(audio_data))
        spectral_energy = np.mean(fft_data[:len(fft_data) // 4])  # Focus on lower frequencies

        return rms > threshold and zcr > 0.005 and spectral_energy > 0.1

    def transcribe_audio_multilingual(self, audio_data, force_language=None):
        """
        Enhanced transcription with proper multi-language support
        """
        try:
            # Normalize audio
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))

            # Prepare audio input
            inputs = self.processor(
                audio_data,
                sampling_rate=self.sample_rate,
                return_tensors="pt"
            )
            input_features = inputs.input_features.to(self.device)

            # Generation parameters optimized for language detection
            generation_kwargs = {
                "max_length": 448,
                "num_beams": 1,
                "do_sample": False,
                "temperature": 0.0,
                "use_cache": True,
                "return_dict_in_generate": True,
                "output_scores": True
            }

            # Handle language forcing
            if force_language:
                # Validate language code
                if force_language in self.language_names:
                    forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                        language=force_language,
                        task="transcribe"
                    )
                    generation_kwargs["forced_decoder_ids"] = forced_decoder_ids
                else:
                    print(f"⚠️ Invalid language code: {force_language}")

            # Generate transcription
            with torch.no_grad():
                outputs = self.model.generate(
                    input_features,
                    **generation_kwargs
                )

            # Decode transcription
            transcription = self.processor.batch_decode(
                outputs.sequences,
                skip_special_tokens=True
            )[0]

            # Extract language information
            detected_language = self.extract_language_from_tokens(outputs.sequences[0])

            # If language detection failed, try content-based detection
            if detected_language == 'Unknown':
                detected_language = self.detect_language_from_content(transcription)

            return transcription.strip(), detected_language

        except Exception as e:
            print(f"⚠️ Transcription error: {e}")
            return None, None

    def extract_language_from_tokens(self, token_sequence):
        """Extract language from Whisper's token sequence"""
        try:
            # Decode the first few tokens to find language markers
            first_tokens = self.processor.tokenizer.decode(token_sequence[:10], skip_special_tokens=False)

            # Look for language tokens in Whisper format <|lang|>
            lang_pattern = r'<\|([a-z]{2})\|>'
            match = re.search(lang_pattern, first_tokens)

            if match:
                lang_code = match.group(1)
                return self.language_names.get(lang_code, lang_code.upper())

            # If no explicit language token found, try to extract from the sequence
            # Convert tokens back to see if we can identify language markers
            token_ids = token_sequence.cpu().numpy()

            # Look for specific language token IDs (these are model-specific)
            # This is a fallback method
            for i, token_id in enumerate(token_ids[:15]):
                token_str = self.processor.tokenizer.decode([token_id])
                if '<|' in token_str and '|>' in token_str:
                    lang_match = re.search(r'<\|([a-z]{2})\|>', token_str)
                    if lang_match:
                        lang_code = lang_match.group(1)
                        return self.language_names.get(lang_code, lang_code.upper())

            return 'Auto-detected'

        except Exception as e:
            print(f"Debug: Token extraction error: {e}")
            return 'Unknown'

    def detect_language_from_content(self, text):
        """Content-based language detection as fallback"""
        if not text or len(text.strip()) < 3:
            return 'Unknown'

        text = text.lower().strip()

        # Script-based detection
        if any('\u0900' <= char <= '\u097F' for char in text):  # Devanagari
            if any(word in text for word in ['है', 'हैं', 'का', 'की', 'को', 'में', 'से']):
                return 'Hindi'
            return 'Devanagari Script'

        if any('\u0600' <= char <= '\u06FF' for char in text):  # Arabic script
            return 'Arabic/Urdu'

        if any('\u4e00' <= char <= '\u9fff' for char in text):  # Chinese characters
            return 'Chinese'

        if any('\u3040' <= char <= '\u309f' for char in text) or any('\u30a0' <= char <= '\u30ff' for char in text):
            return 'Japanese'

        if any('\uac00' <= char <= '\ud7af' for char in text):  # Korean
            return 'Korean'

        # Common word patterns for major languages
        language_patterns = {
            'English': ['the', 'and', 'is', 'are', 'was', 'were', 'have', 'has', 'will', 'would', 'can', 'could',
                        'this', 'that'],
            'Spanish': ['el', 'la', 'de', 'que', 'y', 'es', 'en', 'un', 'ser', 'se', 'no', 'te', 'lo', 'le'],
            'French': ['le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir', 'que', 'pour', 'dans', 'ce'],
            'German': ['der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das', 'mit', 'sich', 'des', 'auf', 'für', 'ist'],
            'Italian': ['il', 'di', 'che', 'è', 'e', 'la', 'per', 'in', 'un', 'da', 'non', 'con', 'del', 'si'],
            'Portuguese': ['de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um', 'para', 'é', 'com', 'não', 'uma'],
            'Russian': ['в', 'и', 'не', 'на', 'я', 'быть', 'он', 'с', 'что', 'а', 'по', 'это', 'она', 'к'],
        }

        max_matches = 0
        detected_lang = 'Unknown'

        for lang, patterns in language_patterns.items():
            matches = sum(1 for pattern in patterns if pattern in text)
            if matches > max_matches:
                max_matches = matches
                detected_lang = lang

        return detected_lang if max_matches > 0 else 'Unknown'

    def show_supported_languages(self):
        """Display supported languages"""
        print("\n🌐 Supported Languages:")
        print("=" * 60)

        # Group by regions
        regions = {
            '🇮🇳 Indian Languages': ['hi', 'bn', 'te', 'ta', 'mr', 'gu', 'kn', 'ml', 'pa', 'or', 'as', 'ne', 'ur', 'sa'],
            '🇪🇺 European Languages': ['en', 'fr', 'es', 'de', 'it', 'pt', 'ru', 'pl', 'nl', 'sv', 'da', 'no', 'fi',
                                      'cs', 'sk', 'hu', 'ro', 'bg', 'hr', 'sr', 'sl', 'et', 'lv', 'lt', 'uk', 'be',
                                      'el', 'ga', 'cy', 'eu', 'ca', 'gl', 'mt', 'is'],
            '🌏 East Asian Languages': ['zh', 'ja', 'ko', 'th', 'vi', 'my', 'mn', 'id', 'ms', 'jw'],
            '🌍 Middle East & Africa': ['ar', 'fa', 'he', 'tr', 'az', 'ka', 'am', 'sw', 'yo', 'zu', 'af', 'so'],
            '🌎 Others': ['sq', 'bs', 'eo', 'kk', 'ky', 'la', 'lb', 'mk', 'mg', 'mi', 'ps', 'gd', 'sn', 'su', 'tg', 'tk',
                         'uz', 'xh', 'yi']
        }

        for region, langs in regions.items():
            print(f"\n{region}:")
            lang_names = [f"{lang}({self.language_names.get(lang, lang)})" for lang in langs if
                          lang in self.language_names]
            # Print in chunks of 4
            for i in range(0, len(lang_names), 4):
                chunk = lang_names[i:i + 4]
                print(f"  {' | '.join(chunk)}")

    def get_supported_languages(self):
        return {
            "en": "English",
            "hi": "Hindi",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "zh": "Chinese",
            "ja": "Japanese"
        }

    def live_transcription(self):
        """Enhanced live transcription with proper multi-language support"""
        print(f"\n🎙️ Multi-Language Live Speech Recognition Active")
        print("🌐 Auto-detecting language from speech")
        print("💡 Commands:")
        print("   - 'exit', 'quit', 'stop' to end")
        print("   - 'show languages' to see supported languages")
        print("   - 'force [lang_code]' to force specific language (e.g., 'force hi' for Hindi)")
        print("   - 'auto' to return to auto-detection mode")
        print("⚡ Processing every 4 seconds for optimal accuracy")
        print("-" * 80)

        consecutive_silence = 0
        max_silence_chunks = 2
        force_language_mode = None

        # Show some common language codes
        print(
            "🔧 Common language codes: en(English), hi(Hindi), es(Spanish), fr(French), de(German), zh(Chinese), ja(Japanese), ar(Arabic)")
        print("-" * 80)

        try:
            while True:
                start_time = time.time()

                # Record audio chunk
                print("🎤 Recording...", end=" ", flush=True)
                audio_chunk = sd.rec(
                    int(self.chunk_duration * self.sample_rate),
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype=np.float32
                )
                sd.wait()
                audio_chunk = audio_chunk.flatten()

                # Speech detection
                if not self.is_speech_detected(audio_chunk):
                    consecutive_silence += 1
                    if consecutive_silence <= max_silence_chunks:
                        print("🔇 Listening for speech...")
                    continue

                consecutive_silence = 0
                print("🎵 Speech detected - Processing...")

                # Transcribe audio
                transcription, detected_language = self.transcribe_audio_multilingual(
                    audio_chunk,
                    force_language=force_language_mode
                )

                processing_time = time.time() - start_time

                if transcription and len(transcription.strip()) > 0:
                    # Display results
                    mode_indicator = f" [FORCED: {force_language_mode}]" if force_language_mode else " [AUTO]"
                    print(f"🌐 [{detected_language}{mode_indicator}] 📝 {transcription}")
                    print(f"⏱️ Processing time: {processing_time:.2f}s")

                    # Check for commands
                    transcription_lower = transcription.lower()

                    # Exit commands
                    if any(cmd in transcription_lower for cmd in ['exit', 'quit', 'stop']):
                        print("👋 Stopping live transcription...")
                        break

                    # Show languages command
                    if 'show languages' in transcription_lower:
                        self.show_supported_languages()
                        continue

                    # Force language commands
                    if transcription_lower.startswith('force '):
                        lang_code = transcription_lower.split('force ')[1].strip()[:2]
                        if lang_code in self.language_names:
                            force_language_mode = lang_code
                            print(f"🔧 Forced to {self.language_names[lang_code]} mode")
                        else:
                            print(f"❌ Invalid language code: {lang_code}")

                    # Auto mode command
                    if 'auto' in transcription_lower and len(transcription_lower) < 10:
                        force_language_mode = None
                        print("🔧 Returned to auto-detection mode")

                else:
                    print("🔄 No clear speech detected...")

                print("-" * 60)

        except KeyboardInterrupt:
            print("\n🛑 Live transcription stopped (Ctrl+C)")
        except Exception as e:
            print(f"❌ Error in live transcription: {e}")


def main():
    print("🚀 Enhanced Multi-Language Live Speech Recognition")
    print("🌐 Supporting 99+ Languages with Auto-Detection")
    print("=" * 80)

    try:
        # Initialize recognizer with tiny model (fastest)
        recognizer = MultiLanguageSpeechRecognizer("openai/whisper-tiny")

        print("\n🎯 Starting Enhanced Live Transcription Mode")
        print("🔊 Make sure your microphone is working and speak clearly")
        print("🌍 The system will automatically detect the language you're speaking")

        # Show supported languages option
        show_langs = input("\n❓ Would you like to see all supported languages? (y/n): ").lower()
        if show_langs == 'y':
            recognizer.show_supported_languages()
            input("\nPress Enter to start live transcription...")

        # Start live transcription
        recognizer.live_transcription()

    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        print("💡 Make sure you have: pip install torch transformers sounddevice")


if __name__ == "__main__":
    main()