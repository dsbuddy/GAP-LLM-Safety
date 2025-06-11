import logging
import warnings

import easyjailbreak
from easyjailbreak.models.model_base import BlackBoxModelBase
import time
import json
import boto3
import backoff


class ClaudeModel(BlackBoxModelBase):
    def __init__(self, model_name: str, generation_config=None):
        """
        Initializes the OpenAI model with necessary parameters.
        :param str model_name: The name of the model to use.
        :param str api_keys: API keys for accessing the OpenAI service.
        :param str template_name: The name of the conversation template, defaults to 'chatgpt'.
        :param dict generation_config: Configuration settings for generation, defaults to an empty dictionary.
        """
        self.client = boto3.client(service_name='bedrock-runtime', region_name="us-west-2") # region_name="us-east-1")
        self.generation_config = generation_config if generation_config is not None else {}
        self.model_name = model_name

    def _generate_request_body(self, prompt):
        # Claude-v3
        if "claude-3" in self.model_name:
            body = json.dumps(
                {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": self.generation_config['max_n_tokens'],
                    "temperature": self.generation_config['temperature'],
                    "top_p": self.generation_config['top_p'],
                    "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                }
            )

        # Claude-v1/2 LLM
        elif "claude" in self.model_name:
            body = json.dumps(
                {
                    "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
                    "max_tokens_to_sample": self.generation_config['max_n_tokens'],
                    "temperature": self.generation_config['temperature'],
                    "top_p": self.generation_config['top_p'],
                    "stop_sequences": ["\n\nHuman:"],
                }
            )

        else:
            raise NotImplementedError(
                f"Bedrock inference for model-id {self.model_name} is not implemented!"
            )

        return body

    def _extract_completion(self, response):
        response_body = json.loads(response.get("body").read())
        print("#########################\nBEDROCK CLAUDE RESPONSE BODY")
        print(response_body)

        if "claude-3" in self.model_name:
            return response_body.get("content")[0]["text"]

        # Claude LLM v1/2
        elif "claude" in self.model_name:
            return response_body.get("completion")

        else:
            raise NotImplementedError(
                f"Bedrock inference for model-id {self.model_name} is not implemented!"
            )

    def generate(self, prompt):
        @backoff.on_exception(
            backoff.expo,
            (
                self.client.exceptions.ServiceQuotaExceededException,
                self.client.exceptions.ModelTimeoutException,
                self.client.exceptions.ModelNotReadyException,
                self.client.exceptions.ThrottlingException,
            ),
        )
        def _submit_request(client, body, model_id):
            accept = "application/json"
            content_type = "application/json"

            response = client.invoke_model(
                body=body, modelId=model_id, accept=accept, contentType=content_type
            )

            return response

        body = self._generate_request_body(prompt)
        response = _submit_request(self.client, body, self.model_name)
        return self._extract_completion(response)

    def batch_generate(self, conversations, **kwargs):
        """
        Generates responses for multiple conversations in a batch.
        :param list[list[str]]|list[str] conversations: A list of conversations, each as a list of messages.
        :return list[str]: A list of responses for each conversation.
        """
        responses = []
        for conversation in conversations:
            if isinstance(conversation, str):
                warnings.warn('For batch generation based on several conversations, provide a list[str] for each conversation. '
                              'Using list[list[str]] will avoid this warning.')
            responses.append(self.generate(conversation, **kwargs))
        return responses