from __future__ import annotations

import openai
import time
from enum import StrEnum
import copy

maximum_thread_count = 61

class RateLimitException(Exception):
    pass

class LLMConnector:
    last_call_timestamp: dict[str, float] = {}
    tokens_used_this_minute: dict[str, int] = {}
    requests_used_this_minute: dict[str, int] = {}
    def __init__(self, model_identifier: str):
        print(f"Connector made for model {model_identifier}")
        self.model_identifier = model_identifier
    
    def _get_token_limit_per_minute(self) -> int|None:
        raise Exception("not implemented yet")

    def _get_request_limit_per_minute(self) -> int|None:
        raise Exception("not implemented yet")
    
    def _get_response(self, request) -> tuple[int, str]:
        raise Exception("not implemented yet")
    
    def _prompt_to_request(self, prompt: str) -> object:
        raise Exception("not implemented yet")
    
    def inject(self, request_text: str, response_text: str) -> None:
        raise Exception("not implemented yet")
    
    def copy(self) -> LLMConnector:
        raise Exception("not implemented yet")

    def ask(self, prompt: str) -> str:
        request = self._prompt_to_request(prompt)
        current = time.time()
        prev = LLMConnector.last_call_timestamp.get(self.model_identifier, None)
        tokens_used = LLMConnector.tokens_used_this_minute.get(self.model_identifier, None)
        requests_used = LLMConnector.requests_used_this_minute.get(self.model_identifier, None)
        est_token_count = len(prompt)//4
        token_limit = self._get_token_limit_per_minute()
        request_limit = self._get_request_limit_per_minute()
        eps_seconds = 1
        if prev is not None and (
            (tokens_used is not None and token_limit is not None and tokens_used + est_token_count > token_limit) or\
            (requests_used is not None and request_limit is not None and requests_used + 1 > request_limit)
        ):
            print(f"Limit reached. Sleeping for: {60 - min(current - prev, 60) + eps_seconds}")
            time.sleep(60 - min(current - prev, 60) + eps_seconds)
            LLMConnector.tokens_used_this_minute[self.model_identifier] = 0
            LLMConnector.requests_used_this_minute[self.model_identifier] = 0
        try:
            tokens_used, response = self._get_response(request)
        except RateLimitException as e:
            print('rate limit exception received. forced sleep.')
            time.sleep(60 - min((current - prev) if prev is not None else 0, 60) + eps_seconds)
            # if we receive this exception a second time, we delegate handling to the upper layer
            tokens_used, response = self._get_response(request)
        current = time.time()
        LLMConnector.tokens_used_this_minute[self.model_identifier] = \
            LLMConnector.tokens_used_this_minute.get(self.model_identifier, 0) + tokens_used
        LLMConnector.requests_used_this_minute[self.model_identifier] = \
            LLMConnector.requests_used_this_minute.get(self.model_identifier, 0) + 1
        LLMConnector.last_call_timestamp[self.model_identifier] = current
        return response
    
class OpenAILib(LLMConnector):
    def get_client(self) -> openai.OpenAI:
        raise Exception(f"Abstract class does not implement get_client")
    
    def __init__(self, model_name: str, system_message: str|None=None, chat_format=True):
        self.model_name = model_name
        self.chat_format = chat_format
        self.chat_log = [] if system_message is None else [{"role": "system", "content": system_message}]
        super().__init__(model_name)

    def copy(self) -> OpenAILib:
        raise Exception(f"Abstract class does not implement copy")
    
    def _get_token_limit_per_minute(self) -> int|None:
        raise Exception(f"Abstract class does not implement _get_token_limit_per_minute")
    
    def _get_request_limit_per_minute(self) -> int|None:
        raise Exception(f"Abstract class does not implement _get_request_limit_per_minute")
    
    def _get_response(self, request) -> tuple[int, str]:
        try:
            response = self.get_client().chat.completions.create(**request)
        except openai.RateLimitError as e:
            import random
            print(f"\n(rate) nClient Exception:\n{e}")
            time.sleep(random.randint(30, 90))
            return 0, ""
        except openai.APITimeoutError as e:
            print(f"\n(timeout) Client Exception:\n{e}")
            time.sleep(10)
            return 0, ""
        except openai.APIStatusError as e:
            print(f"\n(status) Client Exception:\n{e}")
            time.sleep(10)
            return 0, ""
        except Exception as e: # some errors here seem to be from the library. e.g. APIConnectionError.__init__() takes 1 positional argument but 2 were given
            print(f"\nClient Exception:\n{e}")
            time.sleep(10)
            return 0, ""
        tokens_used = int(response.usage.total_tokens)
        response_text = response.choices[0].message.content
        if self.chat_format:
            self.chat_log = request['messages'] + [{'role': 'assistant', 'content': response_text}]
        return tokens_used, response_text
    
    def inject(self, request_text: str, response_text: str):
        # assert self.chat_format, "you can only inject request and responses in a chat format conversation."
        self.chat_log.append({"role": "user", "content": request_text})
        self.chat_log.append({"role": "assistant", "content": response_text})
    
    def _prompt_to_request(self, prompt: str) -> dict:
        # https://platform.openai.com/docs/guides/chat-completions/overview
        return {
                'model': self.model_name,
                # 'model': 'ft:gpt-4o-mini-2024-07-18:personal:ft0-1-100-blindalleys:BmS1GpPI',
                # 'model': 'ft:gpt-4o-mini-2024-07-18:personal:ft0-2-100-blindalleys:BmSuJIgt',
                'model': 'ft:gpt-4o-mini-2024-07-18:personal:ft0-3-300-blindalleys:BmT29Xyu',
                # 'model': 'ft:gpt-4o-mini-2024-07-18:personal:2k-100-blindalleys-miniklondike-spider-50-each:BiUFQWa2',
                # 'model': 'ft:gpt-4o-mini-2024-07-18:personal:minispider-dfsbot-1749143106-train-100-1749146401:Bf9LUd2L', # 100 finetuned model
                # 'model': 'ft:gpt-4o-mini-2024-07-18:personal:minispider-dfsbot-1748995473-train-1748995522:BeWVSTbw', # 1118 finetuned model
                'messages': self.chat_log + [{"role": "user", "content": prompt}],
            }

    def prompt_as_API_request(self, prompt: str, request_id: str):
        request = self._prompt_to_request(prompt)
        return {
            "custom_id": request_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": request,
        }
    
class OpenAIChat(OpenAILib):
    class OpenAIModel(StrEnum):
        GPT_4 = "gpt-4"
        GPT_Turbo_4 = "gpt-4-turbo-2024-04-09" # gpt-4-turbo for latest
        GPT_Turbo_35 = "gpt-3.5-turbo"
        GPT_4O = "gpt-4o"
        GPT_4O_mini = "gpt-4o-mini"
    
    TOKEN_LIMITS = { #https://platform.openai.com/account/limits
        OpenAIModel.GPT_4: 300000,
        OpenAIModel.GPT_Turbo_4: 800000,
        OpenAIModel.GPT_Turbo_35: 10000000,
        OpenAIModel.GPT_4O: 2000000,
        OpenAIModel.GPT_4O_mini: 10000000,
    }
    CLIENT = None
    def __init__(self, openAI_model:OpenAIChat.OpenAIModel, system_message: str|None=None, chat_format=True):
        if openAI_model == OpenAIChat.OpenAIModel.GPT_4O:
            input("[Warning] Using the more expensive GPT 4O model [press Enter to continue]")
        self.openAI_model = openAI_model
        super().__init__(str(self.openAI_model), system_message, chat_format)
    
    # TODO perhaps this should be moved to openAILib, since it has info about the existance of a chat_log
    def copy(self) -> OpenAIChat:
        ret = OpenAIChat(self.openAI_model, None, self.chat_format)
        ret.chat_log = copy.deepcopy(self.chat_log)
        return ret

    def get_client(self):
        from auth import OpenAI_AUTH
        api_key = OpenAI_AUTH
        if OpenAIChat.CLIENT is None:
            OpenAIChat.CLIENT = openai.OpenAI(api_key=api_key)
        return OpenAIChat.CLIENT
    
    def _get_token_limit_per_minute(self) -> int|None:
        return OpenAIChat.TOKEN_LIMITS[self.openAI_model] #TODO share between parallel instances
    
    def _get_request_limit_per_minute(self) -> int|None:
        return None
    
    def as_fine_tuning_example(self):
        messages: list[dict] = copy.deepcopy(self.chat_log)
        for message in messages:
            if message["role"] == "assistant":
                message["weight"] = 0
        assert messages[-1]["role"] == "assistant", "fine-tuning log ends with user message"
        messages[-1]["weight"] = 1
        request = {'messages': messages}
        return request

class GeminiChat(OpenAILib):
    CREDENTIAL_TIMER = None
    class GeminiModel(StrEnum):
        Gemini_1_Pro = "google/gemini-1.0-pro"
        Gemini_15_Pro_002 = "google/gemini-1.5-pro-002"
        Gemini_15_Flash_002 = "google/gemini-1.5-flash-002"
    
    TOKEN_LIMITS = {
        # https://cloud.google.com/vertex-ai/generative-ai/docs/quotas
        # the real limitations here are the requets per minute limits, which has not been implemented here
        GeminiModel.Gemini_1_Pro: 4000000,
        GeminiModel.Gemini_15_Pro_002: 4000000,
        GeminiModel.Gemini_15_Flash_002: 4000000,
    }

    REQUEST_LIMITS = {
        # https://cloud.google.com/vertex-ai/generative-ai/docs/quotas
        # the real limitations here are the requets per minute limits, which has not been implemented here
        GeminiModel.Gemini_1_Pro: 300,
        GeminiModel.Gemini_15_Pro_002: 60,
        GeminiModel.Gemini_15_Flash_002: 200,
    }

    CLIENT = None
    def __init__(self, gemini_model:GeminiChat.GeminiModel, system_message: str|None=None, chat_format=True):
        self.gemini_model = gemini_model
        super().__init__(str(self.gemini_model), system_message, chat_format)

    # TODO perhaps this should be moved to openAILib, since it has info about the existance of a chat_log
    def copy(self) -> GeminiChat:
        ret = GeminiChat(self.gemini_model, None, self.chat_format)
        ret.chat_log = copy.deepcopy(self.chat_log)
        return ret

    def get_client(self):
        expired = False
        if GeminiChat.CREDENTIAL_TIMER is not None and (time.time() - GeminiChat.CREDENTIAL_TIMER) > 30*60:
            expired = True
        if GeminiChat.CLIENT is not None and not expired:
            return GeminiChat.CLIENT
        GeminiChat.CREDENTIAL_TIMER = time.time()
        print("refreshing the credentials")
        import google.auth
        import google.auth.transport.requests
        creds, project = google.auth.default()
        auth_req = google.auth.transport.requests.Request()
        creds.refresh(auth_req) # type: ignore
        from auth import Gemini_Project_ID as project_id
        base_url = f'https://us-central1-aiplatform.googleapis.com/v1beta1/projects/{project_id}/locations/us-central1/endpoints/openapi'
        api_key = creds.token # type: ignore
        GeminiChat.CLIENT = openai.OpenAI(api_key=api_key, base_url=base_url)
        return GeminiChat.CLIENT
    
    def _get_token_limit_per_minute(self) -> int|None:
        #TODO find a better way to access number of connectors sharing this resource
        return max(GeminiChat.TOKEN_LIMITS[self.gemini_model] // maximum_thread_count, 1)
    
    def _get_request_limit_per_minute(self) -> int|None:
        #TODO find a better way to access number of connectors sharing this resource
        return max(GeminiChat.REQUEST_LIMITS[self.gemini_model] // maximum_thread_count, 1)
    
class DeepSeekChat(OpenAILib):
    class DeepSeekModel(StrEnum):
        DEEP_SEEK_CHAT = "deepseek-chat"
        DEEP_SEEK_REASONER = "deepseek-reasoner"
    
    TOKEN_LIMITS = { # no limits: https://api-docs.deepseek.com/quick_start/rate_limit
        DeepSeekModel.DEEP_SEEK_CHAT: 10_000_000,
        DeepSeekModel.DEEP_SEEK_REASONER: 10_000_000
    }
    
    CLIENT = None
    def __init__(self, deepSeek_model:DeepSeekChat.DeepSeekModel, system_message: str|None=None, chat_format=True):
        if deepSeek_model == DeepSeekChat.DeepSeekModel.DEEP_SEEK_REASONER:
            input("[Warning] Using the more expensive deep seek reasoner model [press Enter to continue]")
        self.deepSeek_model = deepSeek_model
        super().__init__(str(self.deepSeek_model), system_message, chat_format)
    
    # TODO perhaps this should be moved to openAILib, since it has info about the existance of a chat_log
    def copy(self) -> DeepSeekChat:
        ret = DeepSeekChat(self.deepSeek_model, None, self.chat_format)
        ret.chat_log = copy.deepcopy(self.chat_log)
        return ret

    def get_client(self):
        from auth import DeepSeek_AUTH
        api_key = DeepSeek_AUTH
        if OpenAIChat.CLIENT is None:
            OpenAIChat.CLIENT = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        return OpenAIChat.CLIENT
    
    def _get_token_limit_per_minute(self) -> int|None:
        return DeepSeekChat.TOKEN_LIMITS[self.deepSeek_model] #TODO share between parallel instances
    
    def _get_request_limit_per_minute(self) -> int|None:
        return None
    
class DeepInfraChat(OpenAILib):
    class DeepInfraModel(StrEnum):
        DEEP_SEEK_R1 = "deepseek-ai/DeepSeek-R1"
    
    TOKEN_LIMITS = { # ?: https://deepinfra.com/docs/advanced/rate-limits
        DeepInfraModel.DEEP_SEEK_R1: 10_000_000,
    }
    
    CLIENT = None
    def __init__(self, deepInfra_model:DeepInfraChat.DeepInfraModel, system_message: str|None=None, chat_format=True):
        if deepInfra_model == DeepInfraChat.DeepInfraModel.DEEP_SEEK_R1:
            input("[Warning] Using the more expensive deep seek reasoner model [press Enter to continue]")
        self.deepInfra_model = deepInfra_model
        super().__init__(str(self.deepInfra_model), system_message, chat_format)
    
    # TODO perhaps this should be moved to openAILib, since it has info about the existance of a chat_log
    def copy(self) -> DeepInfraChat:
        ret = DeepInfraChat(self.deepInfra_model, None, self.chat_format)
        ret.chat_log = copy.deepcopy(self.chat_log)
        return ret

    def get_client(self):
        from auth import DeepInfra_AUTH
        api_key = DeepInfra_AUTH
        if OpenAIChat.CLIENT is None:
            OpenAIChat.CLIENT = openai.OpenAI(api_key=api_key, base_url="https://api.deepinfra.com/v1/openai")
        return OpenAIChat.CLIENT
    
    def _get_token_limit_per_minute(self) -> int|None:
        return DeepInfraChat.TOKEN_LIMITS[self.deepInfra_model] #TODO share between parallel instances
    
    def _get_request_limit_per_minute(self) -> int|None:
        return None