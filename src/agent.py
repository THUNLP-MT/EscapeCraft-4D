import os
import io, base64, httpx, copy, time, re, json, tempfile, queue

from openai import OpenAI
from PIL import Image

from log_config import configure_logger
from config import BASE_URL as _DEFAULT_BASE_URL, API_KEY as _DEFAULT_API_KEY

MODEL_ALIASES = {
    "gemini-3.0-pro-preview": "gemini-3-pro-preview",
    "gemini-3.0-pro": "gemini-3-pro-preview",
}

logger = configure_logger(__name__)

# ==========================================
# Optional SDKs
# ==========================================

# Try to import audio conversion libraries
try:
    from pydub import AudioSegment
    AUDIO_CONVERSION_AVAILABLE = True
except ImportError:
    AUDIO_CONVERSION_AVAILABLE = False
    logger.warning("pydub not available. Audio format conversion may be limited. Install with: pip install pydub")


BASE_URL = os.environ.get("OPENAI_BASE_URL", _DEFAULT_BASE_URL)
API_KEY = os.environ.get("OPENAI_API_KEY", _DEFAULT_API_KEY)
if not API_KEY:
    API_KEY = "sk-placeholder" # Prevent httpx LocalProtocolError if key is missing
retry_flag = True

class AgentPlayer:
    def __init__(self, system_prompt, client=None, model="gpt-4o-2024-08-06", 
                    max_history=None, max_retry=5, history_type="full"):
        """
        arg:
        :history_type: str, default 'full', choose from 'full', 'max', 'key'
        :max_history: int, default None, if history_type is 'max', max_history must to set to a number
        """

        if history_type == "max":
            assert max_history is not None

        logger.info("Initializing the agent.")
        

        original_model = model
        mapped = MODEL_ALIASES.get(model) or MODEL_ALIASES.get((model or "").lower())
        if mapped:
            logger.info(f"Model alias detected: '{model}' -> '{mapped}'")
            model = mapped
        if not client:
            self.client = OpenAI(
                base_url=BASE_URL, 
                api_key=API_KEY,
                http_client=httpx.Client(
                    base_url=BASE_URL,
                    follow_redirects=True,
                ),
            )
            logger.info("Initialized OpenAI client inside AgentPlayer.")
        else:
            self.client = client
            logger.info("Using provided OpenAI client inside AgentPlayer.")
        self.model = model
        self.max_retry = max_retry
        self.history_type = history_type
        self.max_history = max_history

        self.show_transit_prompt = True if history_type == "key" else False 

        self.step_meta_info = [{"key_step": True, "step_prompt": "", "response": ""}] 
        self.last_pos  = 0


        self.notes = []
        
        # 统一使用标准格式的系统消息
        self.system_messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
        ]
        self.interactions = []
        self.key_interactions = []

        self.img_str_pattern = r"data:image\/[a-zA-Z]+;base64,([A-Za-z0-9+/=]+)"

        self._pending_audio_files = []
        self._temp_audio_files = []
        self._last_audio_levels = {}
        self._audio_summary_cache = {}
        self._audio_summary_cooldown = 15.0
        self._data_uri_local_path_map = {}
        self.last_sent_message_snapshot = None

    def add_problem(self, problem, image_path=None, audio_info=None):
        """
        Add a problem to the agent with optional image and audio information.
        
        Args:
            problem: Text description of the problem
            image_path: Path to image file (optional)
            audio_info: Dict with audio information (optional)
                Format: {
                    "trigger_sounds": [{"item_id": str, "description": str}],  # Currently playing trigger sounds
                    "ambient_sounds": [{"item_id": str, "description": str}]   # Currently playing ambient sounds
                }
        """
        content = []
        self.interactions.append({"role": "user", "content": content})

        self.__add_problem(problem)
        if image_path:
            self.__add_image(image_path)
        if audio_info:
            self.__add_audio(audio_info)

    def _build_recordable_message_snapshot(self, message, audio_uri_path_map=None, attach_pending_audio_paths=False):
        """
        Build a JSON-serializable snapshot of the outgoing message list for records.json.
        Replaces base64 media payloads with local file paths when known.
        """
        snapshot = copy.deepcopy(message)
        audio_uri_path_map = audio_uri_path_map or {}

        for msg in snapshot:
            content = msg.get("content")
            if isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    item_type = item.get("type")
                    if item_type == "image_url":
                        image_obj = item.get("image_url", {})
                        url = image_obj.get("url", "")
                        if isinstance(url, str) and url.startswith("data:image/"):
                            local_path = self._data_uri_local_path_map.get(url)
                            image_obj["url"] = local_path or "__base64_image_omitted__"
                    elif item_type == "audio_url":
                        audio_obj = item.get("audio_url", {})
                        url = audio_obj.get("url", "")
                        if isinstance(url, str) and url.startswith("data:audio/"):
                            local_path = audio_uri_path_map.get(url) or self._data_uri_local_path_map.get(url)
                            audio_obj["url"] = local_path or "__base64_audio_omitted__"

        if attach_pending_audio_paths and self._pending_audio_files:
            last_user_msg = None
            for msg in reversed(snapshot):
                if msg.get("role") == "user":
                    last_user_msg = msg
                    break
            if last_user_msg is not None:
                if not isinstance(last_user_msg.get("content"), list):
                    last_user_msg["content"] = [{"type": "text", "text": str(last_user_msg.get("content", ""))}]
                for audio_info in self._pending_audio_files:
                    local_path = audio_info.get("file") or audio_info.get("original_file")
                    if local_path:
                        last_user_msg["content"].append(
                            {
                                "type": "audio_url",
                                "audio_url": {"url": local_path},
                                "_record_only": True,
                                "_audio_meta": {
                                    "item_id": audio_info.get("item_id", "unknown"),
                                    "description": audio_info.get("description", "audio"),
                                    "source_type": audio_info.get("type", "trigger"),
                                },
                            }
                        )

        return snapshot

    def get_last_sent_message_snapshot(self):
        return copy.deepcopy(self.last_sent_message_snapshot)

    def __add_image(self, image_path):
        image = Image.open(image_path)
        if self.show_transit_prompt and len(self.step_meta_info) > 2:
            round_id = len(self.step_meta_info) - 1 
            transition_prompt = f'After {round_id} rounds, the view has become:'
            self.interactions[-1]["content"].append(
                {
                    "type": "text",
                    "text": transition_prompt,
                }
            )

        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
        image_data_uri = f"data:image/jpeg;base64,{base64_image}"
        self._data_uri_local_path_map[image_data_uri] = image_path

        self.interactions[-1]["content"].append(
            {
                "type": "image_url",
                "image_url": {
                    "url": image_data_uri,
                    "detail": "high" if not self.model.startswith("claude") else "auto",
                },
            }
        )
    
    def __add_audio(self, audio_info):
        """
        Enhanced audio attachment:
        - GPT-4o: use text summaries from gpt-4o-audio-preview (no raw audio support)
        - Other models: inline base64 audio via OpenAI-compatible API
        """
        trigger_sounds = audio_info.get("trigger_sounds", [])
        ambient_sounds = audio_info.get("ambient_sounds", [])

        is_gpt4o = self.model.startswith("gpt-4o")

        # Ambient sounds (wind) always converted to text (volume/trend info)
        ambient_texts = self._format_ambient_wind_text(ambient_sounds)
        trigger_summaries = self.interpret_trigger_sounds(audio_info) if is_gpt4o else None

        audio_text_parts = []
        if ambient_texts:
            audio_text_parts.extend(ambient_texts)
        if trigger_summaries:
            # Enhanced prompt: explicitly tell agent that audio hints may contain passwords
            audio_hint_text = "Audio hint (may contain password or important clues): " + " ".join(trigger_summaries)
            audio_hint_text += "\nImportant: If the audio mentions numbers or a password, use it to unlock the exit door by interacting with the door and entering the password in the 'input' field."
            audio_text_parts.append(audio_hint_text)

        if audio_text_parts:
            for text in audio_text_parts:
                self.interactions[-1]["content"].append({"type": "text", "text": text})

        for sound in trigger_sounds + ambient_sounds:
            item_id = sound.get('item_id', 'unknown')
            current_volume = float(sound.get('current_volume', 0.0))
            self._last_audio_levels[item_id] = current_volume

        if is_gpt4o:
            logger.debug("Skipping raw audio upload; using text summaries for gpt-4o.")
            return

        all_sounds = trigger_sounds + ambient_sounds
        self._pending_audio_files.extend([
                {
                    'file': self._prepare_audio_snippet(sound.get('sound_file', '')),
                    'original_file': sound.get('sound_file', ''),
                    'item_id': sound.get('item_id', 'unknown'),
                    'description': sound.get('description', 'playing'),
                    'type': 'trigger' if sound in trigger_sounds else 'ambient'
                }
                for sound in all_sounds if sound.get('sound_file')
            ])
    
    def _convert_audio_to_pcm(self, audio_file_path, max_duration_sec=2):
        """
        Convert any audio file (MP3, WAV, etc.) to a standard WAV format
        (16kHz, mono, 16bit) as required by DashScope and other models.

        Args:
            audio_file_path: Path to audio file
            max_duration_sec: Maximum duration in seconds (default 5)

        Returns:
            Base64 encoded WAV audio string, or None if failed
        """
        if not AUDIO_CONVERSION_AVAILABLE:
            logger.warning("pydub not available, cannot convert audio to WAV format")
            return None
        
        try:
            
            if not os.path.isabs(audio_file_path):
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                audio_file_path = os.path.join(project_root, audio_file_path)
            
            if not os.path.exists(audio_file_path):
                logger.warning(f"Audio file not found: {audio_file_path}")
                return None
            
            audio = AudioSegment.from_file(audio_file_path)

            max_duration_ms = max_duration_sec * 1000
            if len(audio) > max_duration_ms:
                logger.warning(f"Audio file too long ({len(audio)/1000:.1f}s), truncating to {max_duration_sec} seconds")
                audio = audio[:max_duration_ms]

           
            audio = audio.set_frame_rate(16000)  # 16kHz
            audio = audio.set_channels(1)  # Mono
            audio = audio.set_sample_width(2) 

            import io
            wav_io = io.BytesIO()
            audio.export(wav_io, format="wav") 
            wav_bytes = wav_io.getvalue()

            audio_base64 = base64.b64encode(wav_bytes).decode('utf-8')
           
            logger.debug(f"Successfully converted audio to WAV: {audio_file_path} ({len(audio)/1000:.1f}s duration, {len(wav_bytes)} bytes)")

            return audio_base64
        except Exception as e:
            logger.error(f"Failed to convert audio to WAV {audio_file_path}: {e}")
            return None
    
    def _encode_audio_file(self, audio_file_path):
        """
        Encode audio file to base64 string.
        This function now ALWAYS converts to a standard WAV format.
        
        Args:
            audio_file_path: Path to audio file (can be relative or absolute)
            
        Returns:
            Base64 encoded WAV audio string, or None if failed
        """
        try:
            
            if not os.path.isabs(audio_file_path):
                # Use os.path.abspath to resolve relative to CWD, as config.py sets up paths relative to CWD
                audio_file_path = os.path.abspath(audio_file_path)
            
            if not os.path.exists(audio_file_path):
                logger.warning(f"Audio file not found: {audio_file_path}")
                return None
            
            basename = os.path.basename(audio_file_path).lower()
            max_duration_sec = 12 if "radiogram_speech.wav" == basename else 5
            
            return self._convert_audio_to_pcm(audio_file_path, max_duration_sec=max_duration_sec)
            
        except Exception as e:
            logger.error(f"Failed to encode audio file {audio_file_path}: {e}")
            return None

    def _prepare_audio_snippet(self, audio_file_path, max_duration_sec=45):
        """
        Create a short audio snippet suitable for Gemini uploads.
        Returns the path to a temp WAV file or the original path if conversion fails.
        """
        if not audio_file_path:
            return ""

        try:
            if not os.path.isabs(audio_file_path):
                # Use os.path.abspath to resolve relative to CWD, as config.py sets up paths relative to CWD
                audio_file_path = os.path.abspath(audio_file_path)

            if not os.path.exists(audio_file_path):
                logger.warning(f"Audio file not found for snippet: {audio_file_path}")
                return audio_file_path

            if not AUDIO_CONVERSION_AVAILABLE:
                logger.warning("pydub not available, using full audio without trimming.")
                return audio_file_path

            audio = AudioSegment.from_file(audio_file_path)
            max_duration_ms = max_duration_sec * 1000
            if len(audio) > max_duration_ms:
                audio = audio[-max_duration_ms:]

            audio = audio.set_frame_rate(16000)
            audio = audio.set_channels(1)
            audio = audio.set_sample_width(2)

            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            audio.export(tmp_file.name, format="wav")
            self._temp_audio_files.append(tmp_file.name)
            return tmp_file.name
        except Exception as e:
            logger.warning(f"Failed to create audio snippet, falling back to original: {e}")
            return audio_file_path

    def _cleanup_temp_audio_files(self):
        """Remove temporary audio snippets generated for Gemini uploads."""
        for path in self._temp_audio_files:
            try:
                if path and os.path.exists(path):
                    os.remove(path)
            except Exception as e:
                logger.debug(f"Failed to remove temp audio file {path}: {e}")
        self._temp_audio_files.clear()

    def _ask_trigger_sound_to_audio_model(self, audio_file_path):
        """
        Send a trigger sound to gpt-4o-audio-preview and return its semantic summary.
        """
        audio_data = self._encode_audio_file(audio_file_path)
        if not audio_data:
            logger.warning(f"Audio data missing for trigger sound: {audio_file_path}")
            return ""

        try:
            logger.info(f"[AUDIO-PREVIEW] Calling gpt-4o-audio-preview API for: {audio_file_path}")
            logger.debug(f"[AUDIO-PREVIEW] Audio data size: {len(audio_data)} chars (base64)")
            
            completion = self.client.chat.completions.create(
                model="gpt-4o-audio-preview",
                # model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "You are an escape-room audio interpreter. "
                                    "Listen carefully and return a concise hint describing any spoken content, "
                                    "directional clue, or password clue."
                                ),
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            self._build_audio_content_item(audio_data)
                        ],
                    }
                ],
                temperature=0,
            )
            
            logger.info("[AUDIO-PREVIEW] ✓ API call succeeded!")
            
            content = completion.choices[0].message.content
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                result = " ".join(text_parts).strip()
            else:
                result = str(content).strip() if content else ""
            
            logger.info(f"[AUDIO-PREVIEW] Audio summary: {result}")
            return result
            
        except Exception as e:
            logger.error(f"[AUDIO-PREVIEW] ✗ API call failed for {audio_file_path}: {e}")
            import traceback
            logger.debug(f"[AUDIO-PREVIEW] Full traceback:\n{traceback.format_exc()}")
            return ""

    def interpret_trigger_sounds(self, audio_info):
        """
        Filter trigger sounds, call the audio-preview model once per sound, and
        return human-readable summaries for injection into the next prompt.
        """
        trigger_sounds = (audio_info or {}).get("trigger_sounds", [])
        summaries = []
        now = time.time()

        for sound in trigger_sounds:
            sound_file = sound.get("sound_file")
            if not sound_file:
                continue

            cache_key = sound_file
            cached = self._audio_summary_cache.get(cache_key)
            summary_text = ""

            if cached and now - cached["ts"] < self._audio_summary_cooldown:
                summary_text = cached["text"]
                logger.debug(f"Reusing cached audio summary for {cache_key}")
            else:
                summary_text = self._ask_trigger_sound_to_audio_model(sound_file)
                if summary_text:
                    self._audio_summary_cache[cache_key] = {"text": summary_text, "ts": now}

            if summary_text:
                label = sound.get("description") or sound.get("item_id", "trigger")
                summaries.append(f"{label}: {summary_text}")

        return summaries

    def _format_ambient_wind_text(self, ambient_sounds):
        """
        Format ambient (wind) audio readings into textual hints.
        """
        results = []
        for sound in ambient_sounds:
            item_id = sound.get("item_id", "ambient")
            current_volume = float(sound.get("current_volume", 0.0))
            prev_volume = self._last_audio_levels.get(item_id)

            if prev_volume is None:
                trend_word = "steady"
            else:
                diff = current_volume - prev_volume
                if diff > 0.01:
                    trend_word = "increasing"
                elif diff < -0.01:
                    trend_word = "decreasing"
                else:
                    trend_word = "steady"

            results.append(
                f"Wind sound volume: {current_volume:.2f} (trend: {trend_word}). "
                "Louder means you are getting closer to the exit; quieter means farther."
            )
        return results

    def add_response(self, response):
        if self.show_transit_prompt and len(self.step_meta_info) > 1:
            round_id = len(self.step_meta_info) - 1
            response = f'After {round_id} rounds, you got {response}'

        content = [{"type": "text", "text": response}]
        self.interactions.append({"role": "assistant", "content": content})
        self.step_meta_info[-1]['response'] = response

    def __add_problem(self, text):
        if not "<img src='data:image/jpeg;base64," in text:
            self.interactions[-1]["content"].append({"type": "text", "text": text})
        else:
            match = re.search(self.img_str_pattern, text)
            if match:
                img_strs = match.group(1)
                logger.debug("found a img str in desc")
                text_split = text.split(f"<img src='data:image/jpeg;base64,{img_strs}'></img>")
                self.interactions[-1]["content"].append({"type": "text", "text": text_split[0]})
                self.interactions[-1]["content"].append(
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{img_strs}",
                        "detail": "auto"
                        }}
                )
                self.interactions[-1]["content"].append({"type": "text", "text": text_split[1]})
            else:
                self.interactions[-1]["content"].append({"type": "text", "text": text})

    def take_down_note(self, message):
        retry = 0
        tmp_message = copy.deepcopy(message)
        tmp_message.append({"role": "user", "content": [{"type": "text", "text": "Take down your current thought on the lock room and how to escape it based on former information. Use word to describe your it not structured format."}]})
        while retry < self.max_retry:
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=tmp_message,
                    temperature=0,
                )
                logger.debug("Note already taken!")
                return self.notes.append(completion.choices[0].message.content)
            except Exception:
                logger.warning("Error occur while getting response, restarting...")
                retry += 1
                continue
            
        logger.exception("Error occur continuously before max_retry, aborting...")    

    def get_key_interactions(self):
        if self.last_pos == 0:
            self.key_interactions = copy.deepcopy(self.interactions)

        else: 
            if self.step_meta_info[self.last_pos - 1]['key_step']: 
                self.key_interactions.append(self.interactions[self.last_pos*2 - 1])
            
            if self.step_meta_info[self.last_pos]['key_step']: 
                self.key_interactions.append(self.interactions[self.last_pos * 2])
         
        self.last_pos = len(self.step_meta_info)-1

    def get_interactions(self):
        if self.history_type == 'full':
            if self.model.startswith('llama') and len(self.step_meta_info) - 1 >= 22: # context length limits only 22 steps
                message = self.system_messages + self.interactions[-22 * 2:]
            elif (self.model.startswith('gpt') or self.model.startswith('claude'))and len(self.step_meta_info) - 1 >= 50: 
                message = self.system_messages + self.interactions[-50 * 2:]
            else:
                message = self.system_messages + self.interactions
        elif self.history_type == 'key':
            self.get_key_interactions()
            if self.max_history:
                message = self.system_messages + self.key_interactions[-2*self.max_history:]
            else:
                message = self.system_messages + self.key_interactions
        elif self.history_type == 'max':
            if self.max_history is None:
                raise ValueError("max_history must be set when history_type is 'max'")
            message = self.system_messages + self.interactions[-self.max_history * 2:]
        else:
            raise NotImplementedError   
        return message

    def _request_handler(self):
        """Background thread handler for processing model requests."""
        while True:
            try:
                request = self.request_queue.get(timeout=1)
                if request is None:  # Shutdown signal
                    break
                message = request
                response = self._call_model(message)
                self.response_queue.put(response)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in request handler: {e}")
                self.response_queue.put(None)
    
    def _build_audio_content_item(self, audio_base64):
        """
        Generate audio content item based on model type.
        - Gemini models: use input_audio format
        - Other models: use audio_url format
        
        Args:
            audio_base64: Base64 encoded audio string
            
        Returns:
            Dict with appropriate audio content format
        """
        if self.model.startswith("gemini"):
            # Gemini format
            return {
                "type": "input_audio",
                "input_audio": {
                    "data": audio_base64,
                    "format": "wav"
                }
            }
        else:
            # OpenAI-compatible format (audio_url)
            audio_data_uri = f"data:audio/wav;base64,{audio_base64}"
            return {
                "type": "audio_url",
                "audio_url": {"url": audio_data_uri}
            }

    def _log_message_stats(self, message):
        """
        Log statistics about the message content (estimated tokens).
        """
        text_len = 0
        img_count = 0
        audio_count = 0
        audio_duration = 0.0

        # Heuristics for token estimation
        # Text: ~4 chars per token
        # Image: ~765 tokens (OpenAI High Detail standard)
        # Audio: ~25 tokens per second (Gemini standard)
        TOKENS_PER_IMG = 765
        TOKENS_PER_SEC_AUDIO = 25

        for msg in message:
            content = msg.get("content")
            if isinstance(content, str):
                # String content (shouldn't happen with our structure, but handle it)
                text_len += len(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            text_len += len(item.get("text", ""))
                        elif item.get("type") == "image_url" or item.get("type") == "image":
                            img_count += 1
        
        # Count pending audio files
        if hasattr(self, '_pending_audio_files') and self._pending_audio_files:
            audio_count = len(self._pending_audio_files)
            for audio_info in self._pending_audio_files:
                file_path = audio_info.get('file') or audio_info.get('original_file')
                if file_path and os.path.exists(file_path):
                    if AUDIO_CONVERSION_AVAILABLE:
                        try:
                            audio = AudioSegment.from_file(file_path)
                            audio_duration += len(audio) / 1000.0
                        except Exception:
                            pass
        
        est_text_tokens = int(text_len / 4)
        est_img_tokens = img_count * TOKENS_PER_IMG
        est_audio_tokens = int(audio_duration * TOKENS_PER_SEC_AUDIO)
        total_est_tokens = est_text_tokens + est_img_tokens + est_audio_tokens

        logger.info(
            f"[Message Stats] Est. Tokens: ~{total_est_tokens} | "
            f"Text: {text_len} chars (~{est_text_tokens} toks) | "
            f"Images: {img_count} (~{est_img_tokens} toks) | "
            f"Audio: {audio_count} files, {audio_duration:.1f}s (~{est_audio_tokens} toks)"
        )

    def _call_model(self, message):
        """
        Internal method to call the model API.
        """
        self._log_message_stats(message)
        
        if self._pending_audio_files:
            last_user_msg = None
            audio_uri_path_map = {}
            for msg in reversed(message):
                if msg["role"] == "user":
                    last_user_msg = msg
                    break
            
            if last_user_msg:
                logger.info(f"Attaching {len(self._pending_audio_files)} audio files to API request")
                for audio_info in self._pending_audio_files:
                    file_path = audio_info.get('file')
                    if file_path and os.path.exists(file_path):
                        # Encode audio to base64
                        audio_b64 = self._encode_audio_file(file_path)
                        if audio_b64:
                            # Build audio content item based on model type
                            audio_content = self._build_audio_content_item(audio_b64)
                            
                            # For audio_url format, track the mapping
                            if audio_content.get("type") == "audio_url":
                                audio_data_uri = audio_content["audio_url"]["url"]
                                audio_uri_path_map[audio_data_uri] = file_path
                                self._data_uri_local_path_map[audio_data_uri] = file_path
                            
                            last_user_msg["content"].append(audio_content)

            self.last_sent_message_snapshot = self._build_recordable_message_snapshot(
                message,
                audio_uri_path_map=audio_uri_path_map,
                attach_pending_audio_paths=False
            )
             
            self._pending_audio_files = []
            self._cleanup_temp_audio_files()
        else:
            self.last_sent_message_snapshot = self._build_recordable_message_snapshot(
                message,
                attach_pending_audio_paths=False
            )
        
        # 使用标准 OpenAI API 调用
        retry = 0
        while retry < self.max_retry:
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=message,
                    temperature=0,
                )
                logger.debug("Got answer from agent!")
                response_content = completion.choices[0].message.content
                logger.info(f"[RESPONSE] Model: {self.model}\nContent: {response_content}")
                return response_content
            except Exception as e:
                logger.warning(f"Error occur while getting response, receiving: {e}\ncurrent api: {self.client.base_url}\nrestarting...")
                retry += 1
                continue
        
        logger.error("Error occur continuously before max_retry, aborting...")
        return None

    def ask(self):
        """
        Ask the agent for a response.

        Returns:
            Response from the model
        """
        message = self.get_interactions()
        logger.debug("Trying to get answer from agent.")
        return self._call_model(message)
