import oci

CONFIG_PROFILE = "GC3TEST02"

config = oci.config.from_file(
    file_location=r"C:\Users\shshrohi\.oci\config",
    profile_name=CONFIG_PROFILE
)

# ============================================================
# 3. Your real compartment + model
# ============================================================
compartment_id = "ocid1.compartment.oc1..aaaaaaaa2pf2tel6ftytyrdkwaareqpcjfyfit6s62v4qdukfjiflqhlmura"

MODEL_ID = "ocid1.generativeaimodel.oc1.ap-hyderabad-1.amaaaaaask7dceyaaccktjkitpfn3zp3xnkg6yclc6izeahggh2hkwawfjna"

USER_MESSAGE = """
You are a data anonymization expert. Read the following text and replace any 
and all company or customer names that are not 'Oracle' with [Anonymized Customer].

Return only the anonymized text.

Original Text: Oracle is very big company. It works for big giants like Tesla, TATA , Oracle, Google, Microsoft etc.

Anonymized Text:
"""

# ============================================================
# 4. Service Endpoint (must match region → ap-hyderabad-1)
# ============================================================
endpoint = "https://inference.generativeai.ap-hyderabad-1.oci.oraclecloud.com"

# ============================================================
# 5. Create Inference Client
# ============================================================
generative_ai_inference_client = oci.generative_ai_inference.GenerativeAiInferenceClient(
    config=config,
    service_endpoint=endpoint,
    retry_strategy=oci.retry.NoneRetryStrategy(),
    timeout=(10, 240)
)

# ============================================================
# 6. Build Chat Request
# ============================================================
chat_detail = oci.generative_ai_inference.models.ChatDetails()

chat_request = oci.generative_ai_inference.models.CohereChatRequest()
chat_request.message = USER_MESSAGE
chat_request.max_tokens = 4000
chat_request.temperature = 1
chat_request.frequency_penalty = 0
chat_request.top_p = 0.75
chat_request.top_k = 0

# Tell OCI which model to call
chat_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(
    model_id=MODEL_ID
)

chat_detail.chat_request = chat_request
chat_detail.compartment_id = compartment_id

# ============================================================
# 7. Execute Chat
# ============================================================
chat_response = generative_ai_inference_client.chat(chat_detail)

print("**************************Chat Result**************************")
anonymized_text = chat_response.data.chat_response.text
print(anonymized_text)
