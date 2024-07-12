import os
import json
import sys
import boto3

print("impotred successfully...")

prompt="""

        you are a cricket expert now just tell me when RCB will win the IPL?
"""

region_name = 'us-east-1'
bedrock = boto3.client(service_name='bedrock-runtime', region_name=region_name)
payload = {
    "prompt": "[INST]"+prompt+"[/INST]",
    "temperature": 0.5,
    "top_p": 0.9

}

body = json.dumps(payload).encode("utf-8")

model_id = "meta.llama3-8b-instruct-v1:0"

response=bedrock.invoke_model(
    body=body,
    modelId=model_id,
    accept="application/json",
    contentType="application/json"
)


response_body = json.loads(response.get("body").read())
response_text = response_body["generation"]

print(response_text)